#!/usr/bin/env python3
"""
Candidate Screening Tool

This script screens candidates by comparing their resumes against a job description
and identifying the top 3 most qualified candidates using GPT-4o-mini.
"""

import os
import glob
import docx2txt
import openai
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import json
from config import get_api_key

# Load environment variables
load_dotenv()

# Configure OpenAI with custom HTTP client to avoid proxy issues
import httpx

# Get the API key using the same method as generate_profiles.py
openai.api_key = get_api_key()

# Create a custom HTTP client without proxy settings
http_client = httpx.Client(
    timeout=60.0,
    verify=True,
    follow_redirects=True
)

# Initialize OpenAI client with custom HTTP client
client = OpenAI(
    api_key=openai.api_key,
    http_client=http_client
)

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a Word document."""
    try:
        return docx2txt.process(docx_path)
    except Exception as e:
        print(f"Error extracting text from {docx_path}: {e}")
        return ""

def read_job_description(jd_path: str) -> str:
    """Read the job description from a file."""
    try:
        with open(jd_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading job description: {e}")
        return ""

def evaluate_candidate(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Evaluate a single candidate against the job description using GPT-4o-mini."""
    prompt = f"""
    You are an experienced technical recruiter evaluating candidates for a Data Scientist position.
    
    JOB DESCRIPTION:
    {job_description}
    
    CANDIDATE RESUME:
    {resume_text}
    
    Please evaluate this candidate based on the job description and provide:
    1. A score from 1-100 (100 being the best match)
    2. Key strengths that match the job requirements
    3. Any potential concerns or missing qualifications
    
    Return the response as a JSON object with these fields:
    - score (int): 1-100
    - strengths (list of strings)
    - concerns (list of strings)
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that evaluates job candidates based on their resumes and job descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        evaluation = json.loads(response.choices[0].message.content)
        return evaluation
        
    except Exception as e:
        print(f"Error evaluating candidate: {e}")
        return {"score": 0, "strengths": [], "concerns": [f"Evaluation error: {str(e)}"]}

def screen_candidates(resumes_dir: str = None, jd_path: str = None, resume_paths: List[str] = None) -> List[Dict]:
    """
    Screen candidates and return the top matches.
    
    Args:
        resumes_dir: Directory containing resume files (mutually exclusive with resume_paths)
        jd_path: Path to job description file
        resume_paths: List of specific resume files (mutually exclusive with resumes_dir)
    """
    if not jd_path or not os.path.exists(jd_path):
        raise ValueError("Job description file not found or not provided")
    
    # Read job description
    if jd_path.lower().endswith('.docx'):
        job_description = extract_text_from_docx(jd_path)
    else:
        with open(jd_path, 'r', encoding='utf-8') as f:
            job_description = f.read()
    
    if not job_description:
        raise ValueError("Could not read job description")
    
    # Get resume files from either directory or explicit paths
    resume_files = []
    jd_filename = os.path.basename(jd_path).lower()
    
    if resume_paths:
        # Use explicitly provided resume paths, excluding the JD file if it was included
        resume_files = [f for f in resume_paths if os.path.exists(f) and os.path.basename(f).lower() != jd_filename]
    elif resumes_dir and os.path.isdir(resumes_dir):
        # Get all files from directory, excluding the JD file
        resume_files = []
        for ext in ['*.docx', '*.pdf', '*.txt']:
            # Get all files matching the extension
            files = glob.glob(os.path.join(resumes_dir, ext))
            # Filter out the JD file
            files = [f for f in files if os.path.basename(f).lower() != jd_filename]
            resume_files.extend(files)
    
    if not resume_files:
        raise ValueError("No valid resume files found. Please check the provided paths and ensure you're not including the job description file.")
    
    candidates = []
    
    # Process each resume
    for resume_path in resume_files:
        try:
            # Extract text from resume
            if resume_path.lower().endswith('.docx'):
                resume_text = extract_text_from_docx(resume_path)
            else:  # .pdf or .txt
                with open(resume_path, 'r', encoding='utf-8', errors='ignore') as f:
                    resume_text = f.read()
            
            if not resume_text.strip():
                print(f"Warning: Empty or unreadable resume file: {resume_path}")
                continue
            
            # Get candidate name from filename (remove extension, 'resume', and replace _ with space)
            filename = os.path.splitext(os.path.basename(resume_path))[0]
            # Remove 'resume' (case insensitive) and any extra spaces
            candidate_name = ' '.join([word for word in filename.replace('_', ' ').split() 
                                    if word.lower() != 'resume'])
            
            # Evaluate candidate
            evaluation = evaluate_candidate(resume_text, job_description)
            
            candidates.append({
                'name': candidate_name,
                'resume_path': resume_path,
                'evaluation': evaluation
            })
            
            print(f"Processed: {candidate_name} (Score: {evaluation.get('score', 0)}/100)")
            
        except Exception as e:
            print(f"Error processing {resume_path}: {str(e)}")
    
    if not candidates:
        raise ValueError("No valid candidates were processed. Please check the resume files.")
    
    # Sort candidates by score (highest first)
    candidates.sort(key=lambda x: x['evaluation'].get('score', 0), reverse=True)
    
    # Return top candidates (up to 5)
    return candidates[:5]

def generate_screening_report(top_candidates: List[Dict]) -> str:
    """Generate a human-readable screening report."""
    if not top_candidates:
        return "No qualified candidates found."
    
    report = "TOP 3 CANDIDATES\n" + "=" * 50 + "\n\n"
    
    for i, candidate in enumerate(top_candidates, 1):
        eval_data = candidate['evaluation']
        report += f"{i}. {candidate['name']} (Score: {eval_data.get('score', 0)}/100)\n"
        report += "   " + "-" * 47 + "\n"
        
        # Add strengths
        report += "   STRENGTHS:\n"
        for strength in eval_data.get('strengths', [])[:3]:  # Top 3 strengths
            report += f"   • {strength}\n"
        
        # Add concerns if any
        if 'concerns' in eval_data and eval_data['concerns']:
            report += "\n   CONSIDERATIONS:\n"
            for concern in eval_data.get('concerns', [])[:2]:  # Top 2 concerns
                report += f"   • {concern}\n"
        
        report += "\n"
    
    return report

def save_report(report: str, output_file: str = 'screening_report.txt') -> str:
    """Save the screening report to a file."""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    return os.path.abspath(output_file)

def main():
    print("Starting candidate screening...")
    
    # Screen candidates and get top 3
    top_candidates = screen_candidates()
    
    if not top_candidates:
        print("No candidates were evaluated.")
        return
    
    # Generate and save report
    report = generate_screening_report(top_candidates)
    report_path = save_report(report, 'screening_results/screening_report.txt')
    
    print("\n" + "=" * 50)
    print("SCREENING COMPLETE")
    print("=" * 50)
    print(f"\nTop {len(top_candidates)} candidates identified.")
    print(f"Detailed report saved to: {report_path}")
    
    # Print summary
    print("\nTOP CANDIDATES SUMMARY:")
    for i, candidate in enumerate(top_candidates, 1):
        print(f"{i}. {candidate['name']} (Score: {candidate['evaluation'].get('score', 0)}/100)")

if __name__ == "__main__":
    main()
