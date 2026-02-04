"""
Example script demonstrating how to use the interview scheduling functionality.
"""
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from scheduling import InterviewSchedule, InterviewScheduler

# Load environment variables
load_dotenv()

def example_schedule_interview():
    """Example of scheduling an interview and sending reminders."""
    # Create a scheduler instance
    scheduler = InterviewScheduler()
    
    # Create an interview schedule (1 hour from now)
    interview_time = datetime.utcnow() + timedelta(hours=1)
    
    schedule = InterviewSchedule(
        candidate_name="John Doe",
        candidate_email="candidate@example.com",
        interviewer_name="Jane Smith",
        interviewer_email="interviewer@example.com",
        position="Senior Data Scientist",
        interview_type="Technical",
        scheduled_time=interview_time,
        duration_minutes=60,
        meeting_link="https://meet.google.com/xyz-123-abc",
        timezone="America/New_York"
    )
    
    # Schedule the interview
    success, message = scheduler.schedule_interview(schedule)
    print(f"Scheduling result: {success}")
    print(f"Message: {message}")
    
    # Schedule a reminder (24 hours before)
    if success:
        success, message = scheduler.send_reminder(schedule, hours_before=24)
        print(f"\nScheduling reminder: {success}")
        print(f"Message: {message}")

if __name__ == "__main__":
    # Check if SendGrid API key is set
    os.environ["SENDGRID_API_KEY"] = os.getenv("SENDGRID_API_KEY")
    if not os.getenv('SENDGRID_API_KEY'):
        print("Error: SENDGRID_API_KEY environment variable is not set.")
        print("Please add it to your .env file:")
        print("SENDGRID_API_KEY=your_sendgrid_api_key_here")
        print("SENDGRID_FROM_EMAIL=your_email@example.com")
        exit(1)
        
    example_schedule_interview()
