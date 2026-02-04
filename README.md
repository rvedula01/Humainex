# HumAINex: HR Candidate Interview Tracking System
This project generates professional resumes for job candidates, with few of them tailored towards Data Scientist roles and the rest to other software positions. Then uses GenAI capabilities in screening and ranking a resume with reference to the job being applied for. After finding top 3 candidates, the application allows the HR recruiters to setup interview meetings between interviewer and candidates. Email notifications would be immediately sent out to all the members of the meeting. Post-interview, the meeting transcript is evaluated using LLM and subsequent feedback is auto-generated. Dashboards are used to enhance visualizations as needed. 

## Features

- Generates 10 candidate profiles (3 Data Scientists and 7 other technical roles)
- Creates professional Word document resumes for each candidate
- Uses OpenAI's GPT-4o-mini to generate realistic resume content
- Includes diverse skill sets and experiences for each role
- Outputs organized, well-formatted Word documents

## Project Structure
HRCITS/
├── app.py                # Main application entry point
├── generate_profiles.py  # Profile generation script
├── screening.py         # Resume screening module
├── scheduling.py        # Interview scheduling logic
├── interview_feedback.py # Performance analysis module
├── config.py            # Configuration settings
├── .env                 # Environment variables
└── requirements.txt     # Python dependencies

## Prerequisites

- IDE: Windsurf or equivalent vibe coding
- Python 3.8+
- OpenAI API key
- SendGrid API key
- Required Python packages (install using `pip install -r requirements.txt`)

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Run the script:
   ```
   python generate_profiles.py
   ```

2. The script will generate 10 candidate profiles in the `resumes` directory.

## Customization

You can modify the following in `generate_profiles.py`:

- `ROLES` dictionary to add or modify job roles and their requirements
- `generate_candidate_profile()` to change how profiles are generated
- `generate_resume_content()` to modify the resume generation prompt
- `create_word_document()` to change the Word document formatting

## Output

The script creates:
- 10 Word documents (`.docx`) in the `resumes` directory
- Each file is named `[Candidate_Name]_Resume.docx`

## Notes

- Make sure you have sufficient OpenAI API credits before running the script
- The script makes API calls to OpenAI, so an internet connection is required
- Generated resumes are for demonstration purposes only
