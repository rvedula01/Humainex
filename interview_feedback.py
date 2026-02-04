import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import os
from dataclasses import dataclass, asdict
import logging
import PyPDF2
import pdfplumber
from pathlib import Path
import time
import gc
import shutil
from contextlib import contextmanager
import io
import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InterviewFeedback:
    """Standardized feedback structure"""
    candidate_name: str
    interviewer_name: str
    interview_round: str
    role: str
    date: str
    
    # Rating scales (1-5)
    technical_skills: int
    communication: int
    problem_solving: int
    cultural_fit: int
    experience_relevance: int
    overall_rating: int
    
    # Thumbs up/down recommendation
    thumbs_rating: str  # "👍 Thumbs Up", "👎 Thumbs Down", "🤷 Neutral"
    
    # Qualitative feedback
    strengths: str
    areas_for_improvement: str
    specific_examples: str
    recommendation: str  # "Strong Hire", "Hire", "No Hire", "Strong No Hire"
    
    # Comment boxes
    interviewer_notes: str
    follow_up_questions: str
    additional_comments: str  # New general comment box
    concerns_red_flags: str   # New specific concerns box

@dataclass
class ExtractedMetadata:
    """Structure for extracted metadata from PDF"""
    candidate_name: str
    role: str
    interview_round: str
    interviewer_name: str

class PDFTranscriptReader:
    """Utility class for reading PDF transcript files"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str, method: str = "pdfplumber") -> str:
        """Extract text from PDF file"""
        try:
            if method == "pdfplumber":
                return PDFTranscriptReader._extract_with_pdfplumber(pdf_path)
            else:
                return PDFTranscriptReader._extract_with_pypdf2(pdf_path)
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            raise
    
    @staticmethod
    def _extract_with_pdfplumber(pdf_path: str) -> str:
        """Extract text using pdfplumber (better for formatted text)"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    
    @staticmethod
    def _extract_with_pypdf2(pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    @staticmethod
    def batch_extract_from_pdfs(pdf_paths: List[str]) -> Dict[str, str]:
        """Extract text from multiple PDF files"""
        extracted_texts = {}
        for pdf_path in pdf_paths:
            try:
                text = PDFTranscriptReader.extract_text_from_pdf(pdf_path)
                extracted_texts[pdf_path] = text
                logger.info(f"Successfully extracted text from {pdf_path}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {str(e)}")
                extracted_texts[pdf_path] = ""
        return extracted_texts

class InterviewAnalyzer:
    """Main class for analyzing interview transcripts using OpenAI GPT"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        # Create a custom HTTP client with timeout and proxy settings
        http_client = httpx.Client(
            timeout=60.0,
            verify=True,
            follow_redirects=True
        )
        
        # Initialize OpenAI client with custom HTTP client
        self.client = OpenAI(
            api_key=api_key,
            http_client=http_client
        )
        self.model = model
    
    @staticmethod
    def safe_file_operation(operation_func, max_retries=5, delay=0.1):
        """
        Safely perform file operations with retry logic for Windows file locking issues
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Force garbage collection before file operations
                gc.collect()
                
                if attempt > 0:
                    time.sleep(delay * attempt)
                
                return operation_func()
                
            except (PermissionError, OSError) as e:
                last_exception = e
                if attempt == max_retries - 1:
                    logger.error(f"File operation failed after {max_retries} attempts: {e}")
                    raise e
                else:
                    logger.warning(f"File operation attempt {attempt + 1} failed: {e}, retrying...")
        
        raise last_exception

    @staticmethod
    def ensure_file_closed(file_path, max_wait=5):
        """
        Ensure a file is not being used by another process
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                if os.path.exists(file_path):
                    # Try to open file in append mode to check if it's locked
                    with open(file_path, 'a'):
                        pass
                return True
            except (PermissionError, OSError):
                time.sleep(0.1)
                gc.collect()
        
        return False
    
    def extract_metadata_from_transcript(self, transcript: str, filename: str = "") -> ExtractedMetadata:
        """Extract candidate name, role, and interview round from transcript using AI"""
        
        metadata_prompt = f"""
You are an expert at extracting structured information from interview transcripts.

TASK: Extract the following information from the interview transcript:
1. Candidate Name - Look for introductions, names mentioned during the interview
2. Role/Position - The job title or position being interviewed for
3. Interview Round/Type - Type of interview (Technical, HR, Behavioral, Final, L1, L2, etc.)
4. Interviewer Name - Name of the person conducting the interview

FILENAME FOR CONTEXT: {filename}

TRANSCRIPT TO ANALYZE:
{transcript[:3000]}...

INSTRUCTIONS:
- Carefully read through the transcript to find explicit mentions of names, roles, and interview types
- Look for patterns like:
  * "Hi, I'm [name]" or "My name is [name]"
  * "I'm interviewing for [role]" or "applying for [position]"
  * "This is a [type] interview" or "technical round", "HR round", etc.
  * Interviewer introductions like "I'm [name], I'll be conducting..."
- If information is not clearly stated, make reasonable inferences based on:
  * Content discussed (technical questions → Technical Round)
  * Questions asked (behavioral questions → Behavioral Round)
  * Company processes mentioned
- For roles, look for job titles like Software Engineer, Data Scientist, Product Manager, etc.
- For interview rounds, common types include: Technical Round, HR Round, Behavioral Round, System Design, Coding Round, Final Round, L1 Technical, L2 Technical, etc.
- If names are unclear or not mentioned, use "Candidate" and "Interviewer" as fallbacks
- Extract the most specific and relevant information available

Please respond with ONLY a JSON object in this exact format:
{{
    "candidate_name": "Full Name of Candidate or 'Candidate'",
    "role": "Specific Job Title/Position or 'Software Engineer'", 
    "interview_round": "Specific Round Type or 'Technical Round'",
    "interviewer_name": "Interviewer Name or 'Interviewer'"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from interview transcripts. Always respond with valid JSON only. Focus on finding the most accurate information from the actual transcript content."},
                    {"role": "user", "content": metadata_prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                metadata_dict = json.loads(json_str)
                
                return ExtractedMetadata(
                    candidate_name=metadata_dict.get('candidate_name', 'Candidate'),
                    role=metadata_dict.get('role', 'Software Engineer'),
                    interview_round=metadata_dict.get('interview_round', 'Technical Round'),
                    interviewer_name=metadata_dict.get('interviewer_name', 'Interviewer')
                )
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            # Fallback to filename-based extraction
            return self._extract_metadata_from_filename(filename)
    
    def _extract_metadata_from_filename(self, filename: str) -> ExtractedMetadata:
        """Fallback method to extract metadata from filename"""
        try:
            # Remove file extension
            base_name = Path(filename).stem
            
            # Try to parse common filename patterns
            # Examples: "John_Doe_Technical_SoftwareEngineer.pdf"
            #          "NexTurn - L2 Interview - Salesforce Developer - Suraj Awaji.pdf"
            
            if ' - ' in base_name:
                # Handle "Company - Round - Role - Name" pattern
                parts = [part.strip() for part in base_name.split(' - ')]
                if len(parts) >= 4:
                    return ExtractedMetadata(
                        candidate_name=parts[-1],
                        role=parts[-2] if len(parts) > 2 else "Software Engineer",
                        interview_round=parts[1] if 'interview' in parts[1].lower() else parts[1],
                        interviewer_name="Interviewer"
                    )
            
            # Handle underscore-separated format
            parts = base_name.split('_')
            if len(parts) >= 2:
                return ExtractedMetadata(
                    candidate_name=parts[0].replace('_', ' '),
                    role=parts[-1] if len(parts) > 2 else "Software Engineer",
                    interview_round=parts[1] if len(parts) > 1 else "Technical Round",
                    interviewer_name="Interviewer"
                )
            
            # Default fallback
            return ExtractedMetadata(
                candidate_name=base_name.replace('_', ' '),
                role="Software Engineer",
                interview_round="Technical Round",
                interviewer_name="Interviewer"
            )
            
        except Exception as e:
            logger.error(f"Error extracting from filename {filename}: {str(e)}")
            return ExtractedMetadata(
                candidate_name="Candidate",
                role="Software Engineer", 
                interview_round="Technical Round",
                interviewer_name="Interviewer"
            )
        
    def analyze_transcript(self, 
                          transcript: str, 
                          candidate_name: str,
                          interviewer_name: str,
                          interview_round: str,
                          role: str,
                          previous_feedback: Optional[List[Dict]] = None) -> InterviewFeedback:
        """Analyze interview transcript using enhanced prompts"""
        
        analysis_prompt = self._create_enhanced_analysis_prompt(
            transcript, candidate_name, role, interview_round, previous_feedback
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.2,  # Lower for consistent evaluations
                max_tokens=2500
            )
            
            feedback_data = self._parse_gpt_response(response.choices[0].message.content)
            
            feedback = InterviewFeedback(
                candidate_name=candidate_name,
                interviewer_name=interviewer_name,
                interview_round=interview_round,
                role=role,
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **feedback_data
            )
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error analyzing transcript: {str(e)}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Enhanced system prompt for standardized evaluation"""
        return """You are an expert interview evaluator with extensive experience in technical and behavioral interviews. 

EVALUATION PRINCIPLES:
- Provide objective, evidence-based assessments
- Use standardized rating scales consistently
- Focus on demonstrated skills and behaviors from the transcript
- Avoid bias from previous rounds or personal assumptions
- Give balanced feedback with specific examples
- Use professional, constructive language
- Flag any concerns or red flags objectively

RATING CONSISTENCY:
- 5 (Exceptional): Significantly exceeds expectations, outstanding performance
- 4 (Above Average): Clearly above expectations, strong performance  
- 3 (Average): Meets expectations, adequate performance
- 2 (Below Average): Below expectations, some concerns
- 1 (Poor): Well below expectations, major concerns

Provide structured, comprehensive feedback following the exact JSON format requested."""
    
    def _create_enhanced_analysis_prompt(self, 
                                       transcript: str, 
                                       candidate_name: str, 
                                       role: str, 
                                       interview_round: str,
                                       previous_feedback: Optional[List[Dict]] = None) -> str:
        """Enhanced analysis prompt with comprehensive evaluation criteria"""
        
        bias_prevention = ""
        if previous_feedback:
            bias_prevention = f"""
🚨 BIAS PREVENTION NOTICE:
Previous round feedback provided for CONTEXT ONLY. Do NOT let it influence your independent assessment.
Evaluate this interview round completely independently based solely on this transcript.

Previous Context (for reference only): {str(previous_feedback)}
"""
        
        return f"""
INTERVIEW EVALUATION REQUEST

📋 INTERVIEW DETAILS:
• Candidate: {candidate_name}
• Role: {role}  
• Round: {interview_round}
• Evaluation Type: Standardized Comprehensive Assessment

{bias_prevention}

📝 TRANSCRIPT TO ANALYZE:
{transcript}

🎯 EVALUATION REQUIREMENTS:

Please provide a comprehensive evaluation in the following JSON format:

{{
    "technical_skills": [1-5 rating],
    "communication": [1-5 rating], 
    "problem_solving": [1-5 rating],
    "cultural_fit": [1-5 rating],
    "experience_relevance": [1-5 rating],
    "overall_rating": [1-5 rating],
    "thumbs_rating": "[👍 Thumbs Up/👎 Thumbs Down/🤷 Neutral]",
    "strengths": "[List 3-5 key strengths with specific examples]",
    "areas_for_improvement": "[List 2-4 areas needing development]", 
    "specific_examples": "[Quote specific moments from interview that support your ratings]",
    "recommendation": "[Strong Hire/Hire/No Hire/Strong No Hire]",
    "interviewer_notes": "[Key observations and insights for hiring team]",
    "follow_up_questions": "[Suggested questions for next rounds or areas to probe deeper]",
    "additional_comments": "[Any other relevant observations or context]",
    "concerns_red_flags": "[Any concerns, inconsistencies, or red flags observed]"
}}

🔍 EVALUATION FOCUS AREAS:

TECHNICAL SKILLS ({interview_round}):
- Depth of knowledge relevant to {role}
- Problem-solving approach and methodology
- Code quality, architecture thinking (if applicable)
- Understanding of best practices and industry standards

COMMUNICATION:
- Clarity of explanations and thought process
- Active listening and question comprehension  
- Professional demeanor and confidence
- Ability to explain complex concepts simply

PROBLEM SOLVING:
- Structured thinking and approach
- Creativity and innovation in solutions
- Handling of ambiguity and edge cases
- Learning agility when facing new problems

CULTURAL FIT:
- Alignment with company values and culture
- Team collaboration potential
- Growth mindset and adaptability
- Professional attitude and work ethic

EXPERIENCE RELEVANCE:
- Relevance of past experience to {role}
- Demonstrated impact and achievements
- Leadership and initiative examples
- Career progression and learning curve

📊 RATING SCALE GUIDE:
5 - Exceptional: Significantly exceeds expectations, top 10% of candidates
4 - Above Average: Clearly above expectations, strong performer
3 - Average: Meets expectations, solid candidate  
2 - Below Average: Below expectations, has concerns
1 - Poor: Well below expectations, major red flags

🎯 THUMBS RATING GUIDE:
👍 Thumbs Up: Strong positive recommendation, confident hire
🤷 Neutral: Mixed signals, needs discussion or additional rounds
👎 Thumbs Down: Not recommended, significant concerns

⚠️ IMPORTANT GUIDELINES:
- Base ALL ratings on evidence from this transcript only
- Provide specific quotes and examples to support ratings
- Be objective and avoid personal bias
- Consider the seniority level expected for {role}
- Flag any inconsistencies or concerning responses
- Use professional, constructive language
- Consider both technical competence and cultural alignment
"""
    
    def analyze_pdf_transcript(self,
                              pdf_path: str,
                              previous_feedback: Optional[List[Dict]] = None) -> InterviewFeedback:
        """Analyze PDF transcript file with automatic metadata extraction"""
        try:
            # Extract text from PDF
            transcript = PDFTranscriptReader.extract_text_from_pdf(pdf_path)
            
            # Extract metadata from transcript using AI
            filename = Path(pdf_path).name
            metadata = self.extract_metadata_from_transcript(transcript, filename)
            
            logger.info(f"Extracted metadata - Name: {metadata.candidate_name}, Role: {metadata.role}, Round: {metadata.interview_round}")
            
            # Analyze transcript with extracted metadata
            return self.analyze_transcript(
                transcript=transcript,
                candidate_name=metadata.candidate_name,
                interviewer_name=metadata.interviewer_name,
                interview_round=metadata.interview_round,
                role=metadata.role,
                previous_feedback=previous_feedback
            )
            
        except Exception as e:
            logger.error(f"Error analyzing PDF {pdf_path}: {str(e)}")
            raise
    
    def _parse_gpt_response(self, response_text: str) -> Dict:
        """Parse GPT response and extract structured data"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return self._create_default_feedback()
    
    def _create_default_feedback(self) -> Dict:
        """Create default feedback structure if parsing fails"""
        return {
            "technical_skills": 3, "communication": 3, "problem_solving": 3,
            "cultural_fit": 3, "experience_relevance": 3, "overall_rating": 3,
            "thumbs_rating": "🤷 Neutral",
            "strengths": "Analysis failed - manual review required",
            "areas_for_improvement": "Analysis failed - manual review required", 
            "specific_examples": "Analysis failed - manual review required",
            "recommendation": "Manual Review Required",
            "interviewer_notes": "Automated analysis failed - please review manually",
            "follow_up_questions": "Manual review required",
            "additional_comments": "System error occurred during analysis",
            "concerns_red_flags": "Unable to assess - manual review needed"
        }
    
    def batch_analyze_pdf_transcripts(self, pdf_files: List[str]) -> List[InterviewFeedback]:
        """Analyze multiple PDF transcript files in batch with automatic metadata extraction"""
        feedbacks = []
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                # Get previous feedback for bias prevention
                previous_feedback = None
                if i > 0:
                    previous_feedback = [asdict(fb) for fb in feedbacks[-2:]]
                
                feedback = self.analyze_pdf_transcript(
                    pdf_path=pdf_path,
                    previous_feedback=previous_feedback
                )
                
                feedbacks.append(feedback)
                logger.info(f"Analyzed PDF {i+1}/{len(pdf_files)}: {Path(pdf_path).name}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {pdf_path}: {str(e)}")
                continue
        
        return feedbacks
    
    def export_feedback_to_excel_streamlit_safe(self, feedbacks: List[InterviewFeedback]) -> bytes:
        """Streamlit-safe Excel export that returns bytes for download - NO TEMP FILES"""
        try:
            # Prepare data
            data = [asdict(feedback) for feedback in feedbacks]
            df = pd.DataFrame(data)
            
            # Create Excel file in memory
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main feedback data
                df.to_excel(writer, sheet_name='Interview_Feedback', index=False)
                
                # Summary statistics
                try:
                    summary_stats = self._create_summary_stats(df)
                    summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=True)
                except Exception as stats_error:
                    logger.warning(f"Could not create summary stats: {stats_error}")
                
                # Ratings analysis
                try:
                    ratings_columns = ['candidate_name', 'interview_round', 'technical_skills', 
                                    'communication', 'problem_solving', 'cultural_fit', 
                                    'experience_relevance', 'overall_rating', 'thumbs_rating', 'recommendation']
                    
                    available_columns = [col for col in ratings_columns if col in df.columns]
                    if available_columns:
                        ratings_df = df[available_columns]
                        ratings_df.to_excel(writer, sheet_name='Ratings_Analysis', index=False)
                except Exception as ratings_error:
                    logger.warning(f"Could not create ratings analysis: {ratings_error}")
            
            # Get the bytes
            output.seek(0)
            excel_bytes = output.getvalue()
            output.close()
            
            logger.info("Excel file successfully created in memory")
            return excel_bytes
            
        except Exception as e:
            logger.error(f"Error creating Excel bytes: {str(e)}")
            return None

    def generate_blind_feedback_report_bytes(self, feedbacks: List[InterviewFeedback]) -> bytes:
        """Generate blind feedback report in memory and return bytes"""
        try:
            blind_feedback = []
            
            for i, feedback in enumerate(feedbacks):
                blind_entry = {
                    'interview_id': f"INT_{i+1:03d}",
                    'round': feedback.interview_round,
                    'role': feedback.role,
                    'technical_skills': feedback.technical_skills,
                    'communication': feedback.communication,
                    'problem_solving': feedback.problem_solving, 
                    'cultural_fit': feedback.cultural_fit,
                    'experience_relevance': feedback.experience_relevance,
                    'overall_rating': feedback.overall_rating,
                    'thumbs_rating': feedback.thumbs_rating,
                    'recommendation': feedback.recommendation,
                    'strengths': feedback.strengths,
                    'areas_for_improvement': feedback.areas_for_improvement,
                    'concerns_red_flags': feedback.concerns_red_flags
                }
                blind_feedback.append(blind_entry)
            
            df_blind = pd.DataFrame(blind_feedback)
            
            # Create Excel in memory
            output = io.BytesIO()
            df_blind.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            excel_bytes = output.getvalue()
            output.close()
            
            logger.info("Blind feedback report created in memory")
            return excel_bytes
            
        except Exception as e:
            logger.error(f"Failed to generate blind feedback report: {str(e)}")
            return None

    def export_feedback_to_excel(self, feedbacks: List[InterviewFeedback], output_file: str):
        """Enhanced Excel export with robust file handling and Windows compatibility"""
        @contextmanager
        def safe_temporary_file(suffix='.xlsx'):
            """Context manager for safe temporary file handling"""
            import tempfile
            
            # Create temp file with more unique name
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            temp_dir = tempfile.gettempdir()
            temp_filename = f"interview_feedback_{timestamp}_{os.getpid()}{suffix}"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            try:
                yield temp_path
            finally:
                # Force cleanup with multiple attempts
                for attempt in range(5):
                    try:
                        if os.path.exists(temp_path):
                            # Force garbage collection
                            gc.collect()
                            time.sleep(0.1)  # Small delay
                            os.unlink(temp_path)
                            break
                    except (PermissionError, OSError) as e:
                        if attempt == 4:  # Last attempt
                            logger.warning(f"Could not delete temp file {temp_path}: {e}")
                        else:
                            time.sleep(0.2 * (attempt + 1))  # Increasing delay

        try:
            # Prepare data
            data = [asdict(feedback) for feedback in feedbacks]
            df = pd.DataFrame(data)
            
            # Handle existing output file
            original_output = output_file
            if os.path.exists(output_file):
                try:
                    # Test if file is accessible
                    with open(output_file, 'r+b'):
                        pass
                except (PermissionError, OSError):
                    # File is locked, create new filename
                    base, ext = os.path.splitext(output_file)
                    timestamp = int(time.time())
                    output_file = f"{base}_{timestamp}{ext}"
                    logger.info(f"Original file locked, using: {output_file}")
            
            # Use safe temporary file context manager
            with safe_temporary_file('.xlsx') as temp_path:
                # Create Excel file in temp location first
                try:
                    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
                        # Main feedback data
                        df.to_excel(writer, sheet_name='Interview_Feedback', index=False)
                        
                        # Summary statistics
                        try:
                            summary_stats = self._create_summary_stats(df)
                            summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=True)
                        except Exception as stats_error:
                            logger.warning(f"Could not create summary stats: {stats_error}")
                        
                        # Ratings analysis
                        try:
                            ratings_columns = ['candidate_name', 'interview_round', 'technical_skills', 
                                            'communication', 'problem_solving', 'cultural_fit', 
                                            'experience_relevance', 'overall_rating', 'thumbs_rating', 'recommendation']
                            
                            available_columns = [col for col in ratings_columns if col in df.columns]
                            if available_columns:
                                ratings_df = df[available_columns]
                                ratings_df.to_excel(writer, sheet_name='Ratings_Analysis', index=False)
                        except Exception as ratings_error:
                            logger.warning(f"Could not create ratings analysis: {ratings_error}")
                    
                    # Ensure writer is properly closed before copying
                    writer = None
                    gc.collect()
                    time.sleep(0.1)
                    
                    # Copy from temp to final location
                    shutil.copy2(temp_path, output_file)
                    
                    logger.info(f"Excel file successfully created: {output_file}")
                    print(f"✅ Excel file successfully created: {output_file}")
                    
                    if output_file != original_output:
                        print(f"📝 Note: Original filename was in use, saved as: {os.path.basename(output_file)}")
                    
                    return True
                    
                except Exception as write_error:
                    logger.error(f"Error writing Excel file: {write_error}")
                    return self._fallback_csv_export(df, output_file)
                    
        except Exception as e:
            logger.error(f"Error in export_feedback_to_excel: {str(e)}")
            return self._fallback_csv_export(df if 'df' in locals() else pd.DataFrame(), output_file)

    def _fallback_csv_export(self, df: pd.DataFrame, output_file: str) -> bool:
        """Enhanced fallback method to export as CSV if Excel fails"""
        try:
            # Generate unique CSV filename
            csv_file = output_file.replace('.xlsx', '.csv').replace('.xls', '.csv')
            
            if os.path.exists(csv_file):
                base, ext = os.path.splitext(csv_file)
                timestamp = int(time.time())
                csv_file = f"{base}_{timestamp}{ext}"
            
            # Direct CSV export with error handling
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"Exported as CSV: {csv_file}")
            print(f"⚠️ Excel export failed, saved as CSV: {csv_file}")
            print(f"💡 You can open the CSV file in Excel manually")
            return True
            
        except Exception as csv_error:
            logger.error(f"CSV export also failed: {csv_error}")
            print(f"❌ Both Excel and CSV export failed: {csv_error}")
            return False

    def export_feedback_robust(self, feedbacks: List[InterviewFeedback], output_file: str):
        """
        Most robust export method with multiple fallback strategies
        """
        
        def excel_export_operation():
            """Inner function for the actual Excel export"""
            # Prepare data
            data = [asdict(feedback) for feedback in feedbacks]
            df = pd.DataFrame(data)
            
            # Generate unique output filename if needed
            final_output = output_file
            counter = 1
            while os.path.exists(final_output) and not self.ensure_file_closed(final_output):
                base, ext = os.path.splitext(output_file)
                final_output = f"{base}_{counter}{ext}"
                counter += 1
            
            # Direct write without temporary files
            with pd.ExcelWriter(final_output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Interview_Feedback', index=False)
                
                try:
                    summary_stats = self._create_summary_stats(df)
                    summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=True)
                except Exception as e:
                    logger.warning(f"Could not create summary stats: {e}")
            
            return final_output
        
        try:
            # Try the main Excel export with retry logic
            result_file = self.safe_file_operation(excel_export_operation)
            
            logger.info(f"Excel file successfully created: {result_file}")
            print(f"✅ Excel file successfully created: {result_file}")
            
            if result_file != output_file:
                print(f"📝 Note: Used alternative filename: {os.path.basename(result_file)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            
            # Fallback to CSV
            try:
                data = [asdict(feedback) for feedback in feedbacks]
                df = pd.DataFrame(data)
                
                csv_file = output_file.replace('.xlsx', '.csv')
                if os.path.exists(csv_file):
                    base, ext = os.path.splitext(csv_file)
                    csv_file = f"{base}_{int(time.time())}{ext}"
                
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                
                logger.info(f"Exported as CSV fallback: {csv_file}")
                print(f"⚠️ Excel failed, saved as CSV: {csv_file}")
                return True
                
            except Exception as csv_error:
                logger.error(f"Both Excel and CSV export failed: {csv_error}")
                print(f"❌ All export methods failed: {csv_error}")
                return False

    def _create_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics with error handling"""
        try:
            numeric_columns = ['technical_skills', 'communication', 'problem_solving', 
                            'cultural_fit', 'experience_relevance', 'overall_rating']
            
            # Only include columns that exist and have numeric data
            available_numeric_columns = []
            for col in numeric_columns:
                if col in df.columns:
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        available_numeric_columns.append(col)
                    except (ValueError, TypeError):
                        logger.warning(f"Column {col} is not numeric, skipping from summary stats")
            
            if available_numeric_columns:
                return df[available_numeric_columns].describe()
            else:
                # Return basic info if no numeric columns available
                return pd.DataFrame({
                    'Total Interviews': [len(df)],
                    'Candidates': [df['candidate_name'].nunique() if 'candidate_name' in df.columns else 'N/A'],
                    'Interview Rounds': [df['interview_round'].nunique() if 'interview_round' in df.columns else 'N/A']
                })
                
        except Exception as e:
            logger.error(f"Error creating summary stats: {e}")
            return pd.DataFrame({'Error': ['Could not generate summary statistics']})

    def generate_blind_feedback_report(self, feedbacks: List[InterviewFeedback], output_file: str):
        """Generate blind feedback report for unbiased collaboration"""
        try:
            blind_feedback = []
            
            for i, feedback in enumerate(feedbacks):
                blind_entry = {
                    'interview_id': f"INT_{i+1:03d}",
                    'round': feedback.interview_round,
                    'role': feedback.role,
                    'technical_skills': feedback.technical_skills,
                    'communication': feedback.communication,
                    'problem_solving': feedback.problem_solving, 
                    'cultural_fit': feedback.cultural_fit,
                    'experience_relevance': feedback.experience_relevance,
                    'overall_rating': feedback.overall_rating,
                    'thumbs_rating': feedback.thumbs_rating,
                    'recommendation': feedback.recommendation,
                    'strengths': feedback.strengths,
                    'areas_for_improvement': feedback.areas_for_improvement,
                    'concerns_red_flags': feedback.concerns_red_flags
                }
                blind_feedback.append(blind_entry)
            
            df_blind = pd.DataFrame(blind_feedback)
            df_blind.to_excel(output_file, index=False)
            
            logger.info(f"Blind feedback report exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate blind feedback report: {str(e)}")
            raise

# Example usage
def main():
    """Example usage of the enhanced InterviewAnalyzer"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    analyzer = InterviewAnalyzer(api_key)
    
    # Single PDF analysis with automatic metadata extraction
    try:
        pdf_path = r"hrcits\NexTurn - L2 Interview - Salesforce Developer - Suraj Awaji.pdf"
        
        if os.path.exists(pdf_path):
            feedback = analyzer.analyze_pdf_transcript(pdf_path=pdf_path)
            
            print("✅ Analysis completed successfully!")
            print(f"📊 Candidate: {feedback.candidate_name}")
            print(f"🎯 Role: {feedback.role}")
            print(f"📋 Round: {feedback.interview_round}")
            print(f"📊 Overall Rating: {feedback.overall_rating}/5")
            print(f"👍 Thumbs Rating: {feedback.thumbs_rating}") 
            print(f"🎯 Recommendation: {feedback.recommendation}")
            print(f"💪 Key Strengths: {feedback.strengths[:100]}...")
            
            analyzer.export_feedback_to_excel([feedback], "enhanced_interview_feedback.xlsx")
        else:
            print(f"❌ PDF file not found: {pdf_path}")
    
    except Exception as e:
        print(f"❌ Error analyzing PDF: {str(e)}")

def analyze_pdf_folder(folder_path: str, output_prefix: str = "interview_analysis"):
    """Analyze all PDF files in a folder with automatic metadata extraction"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    analyzer = InterviewAnalyzer(api_key)
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    try:
        feedbacks = analyzer.batch_analyze_pdf_transcripts([str(pdf) for pdf in pdf_files])
        
        analyzer.export_feedback_to_excel(feedbacks, f"{output_prefix}_enhanced_feedback.xlsx")
        analyzer.generate_blind_feedback_report(feedbacks, f"{output_prefix}_blind_feedback.xlsx")
        
        print(f"✅ Enhanced analysis complete! Results saved to {output_prefix}_enhanced_feedback.xlsx")
        
    except Exception as e:
        print(f"❌ Error analyzing folder: {str(e)}")

if __name__ == "__main__":
    main()