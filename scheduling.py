"""
Interview Scheduling Module

Handles interview scheduling and email notifications using SendGrid.
"""

import os
import datetime
from dataclasses import dataclass
from typing import Optional
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv
import pytz
import uuid

# Load environment variables
load_dotenv()

@dataclass
class InterviewSchedule:
    """Data class for interview scheduling."""
    candidate_name: str
    candidate_email: str
    interviewer_name: str
    interviewer_email: str
    position: str
    interview_type: str
    scheduled_time: datetime.datetime
    duration_minutes: int = 60
    meeting_link: Optional[str] = None
    timezone: str = 'UTC'
    interview_id: str = None
    
    def __post_init__(self):
        if not self.interview_id:
            self.interview_id = str(uuid.uuid4())

class InterviewScheduler:
    """Handles interview scheduling and notifications."""
    
    def __init__(self, sendgrid_api_key: str = None):
        """Initialize with optional SendGrid API key."""
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.email_enabled = bool(self.sendgrid_api_key)
        if self.email_enabled:
            try:
                self.sg = SendGridAPIClient(api_key=self.sendgrid_api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize SendGrid client: {e}")
                self.email_enabled = False
    
    def schedule_interview(self, schedule: InterviewSchedule) -> tuple[bool, str]:
        """Schedule interview and optionally send notifications if email is enabled."""
        try:
            print(f"[DEBUG] Starting interview scheduling for {schedule.candidate_name}")
            
            # Format the interview time in the specified timezone
            try:
                tz = pytz.timezone(schedule.timezone)
                local_time = schedule.scheduled_time.astimezone(tz)
                formatted_time = local_time.strftime('%A, %B %d, %Y at %I:%M %p %Z')
                print(f"[DEBUG] Formatted time: {formatted_time}")
            except Exception as tz_error:
                print(f"[ERROR] Timezone error: {tz_error}")
                raise
            
            # Generate a meeting link if not provided
            if not schedule.meeting_link:
                schedule.meeting_link = f"https://meet.google.com/xyz-{schedule.interview_id[:8]}"
            print(f"[DEBUG] Meeting link: {schedule.meeting_link}")
            
            # Only send emails if email is enabled
            if self.email_enabled:
                print("[DEBUG] Email is enabled, preparing to send notifications")
                try:
                    print(f"[DEBUG] Sending email to candidate: {schedule.candidate_email}")
                    # Send email to candidate
                    candidate_success = self._send_email(
                        to_email=schedule.candidate_email,
                        to_name=schedule.candidate_name,
                        schedule=schedule,
                        is_candidate=True
                    )
                    
                    print(f"[DEBUG] Sending email to interviewer: {schedule.interviewer_email}")
                    # Send email to interviewer
                    interviewer_success = self._send_email(
                        to_email=schedule.interviewer_email,
                        to_name=schedule.interviewer_name,
                        schedule=schedule,
                        is_candidate=False
                    )
                    
                    if candidate_success and interviewer_success:
                        print("[DEBUG] Both emails sent successfully")
                        return True, "Interview scheduled and notifications sent successfully"
                    else:
                        print(f"[WARNING] Email sending partially failed - candidate: {candidate_success}, interviewer: {interviewer_success}")
                        return True, "Interview scheduled but some email notifications may have failed"
                        
                except Exception as email_error:
                    # If email fails, still complete the scheduling but log the error
                    print(f"[ERROR] Failed to send email notifications: {str(email_error)}")
                    print("[DEBUG] Email error details:", str(email_error))
                    import traceback
                    traceback.print_exc()
                    return True, "Interview scheduled but email notifications failed to send"
            else:
                print("[DEBUG] Email is disabled, skipping email notifications")
                return True, "Interview scheduled (email notifications disabled)"
                
        except Exception as e:
            print(f"[ERROR] Error in schedule_interview: {str(e)}")
            print("[DEBUG] Full error details:", str(e))
            import traceback
            traceback.print_exc()
            return False, f"Error scheduling interview: {str(e)}"
    
    def send_reminder(self, schedule: InterviewSchedule, hours_before: int = 24) -> tuple[bool, str]:
        """Send reminder email for scheduled interview."""
        try:
            reminder_time = schedule.scheduled_time - datetime.timedelta(hours=hours_before)
            if datetime.datetime.now(pytz.utc) >= schedule.scheduled_time:
                return False, "Cannot send reminder for past interviews"
                
            # Send reminder to candidate
            self._send_email(
                to_email=schedule.candidate_email,
                to_name=schedule.candidate_name,
                schedule=schedule,
                is_candidate=True,
                is_reminder=True
            )
            
            # Send reminder to interviewer
            self._send_email(
                to_email=schedule.interviewer_email,
                to_name=schedule.interviewer_name,
                schedule=schedule,
                is_candidate=False,
                is_reminder=True
            )
            
            return True, "Reminder sent successfully"
            
        except Exception as e:
            return False, f"Error sending reminder: {str(e)}"
    
    def _send_email(self, to_email: str, to_name: str, schedule: InterviewSchedule, 
                   is_candidate: bool, is_reminder: bool = False) -> bool:
        """Send email notification for interview."""
        if not hasattr(self, 'sg') or not self.email_enabled:
            return False
            
        try:
            # Get timezone and format time
            try:
                tz = pytz.timezone(schedule.timezone)
            except pytz.exceptions.UnknownTimeZoneError:
                tz = pytz.UTC
                
            local_time = schedule.scheduled_time.astimezone(tz)
            time_str = local_time.strftime("%A, %B %d, %Y at %I:%M %p %Z")
            
            # Email content
            subject = "Reminder: " if is_reminder else ""
            subject += f"Interview Invitation - {schedule.position}"
            
            if is_candidate:
                greeting = f"Dear {to_name},"
                intro = "This is a reminder about your upcoming interview." if is_reminder else \
                       f"We are pleased to invite you for a {schedule.interview_type} interview."
            else:
                greeting = f"Hello {to_name},"
                intro = "Reminder: You have an upcoming interview." if is_reminder else \
                       f"You have been scheduled to interview {schedule.candidate_name}."
            
            # Create email content
            meeting_link_html = f""
            if schedule.meeting_link:
                meeting_link_html = f"<li><strong>Candidate & Meeting Details:</strong> <a href='{schedule.meeting_link}'>{schedule.meeting_link}</a></li>"
            
            html_content = f"""
            <html>
            <body>
                <p>{greeting}</p>
                <p>{intro}</p>
                <p><strong>Interview Details:</strong></p>
                <ul>
                    <li><strong>Date & Time:</strong> {time_str}</li>
                    <li><strong>Duration:</strong> {schedule.duration_minutes} minutes</li>
                    <li><strong>Type:</strong> {schedule.interview_type}</li>
                    <li><strong>Position:</strong> {schedule.position}</li>
                    {meeting_link_html}
                </ul>
                <p>Best regards,<br>HR Team</p>
            </body>
            </html>
            """.format(
                greeting=greeting,
                intro=intro,
                time_str=time_str,
                duration=schedule.duration_minutes,
                interview_type=schedule.interview_type,
                position=schedule.position,
                meeting_link=meeting_link_html
            )
            
            # Create email message
            message = Mail(
                from_email=os.getenv('SENDGRID_FROM_EMAIL', 'noreply@example.com'),
                to_emails=to_email,
                subject=subject,
                html_content=html_content.strip()
            )
            
            # Send email
            response = self.sg.send(message)
            return response.status_code in [200, 202]
            
        except Exception as e:
            print(f"Error sending email to {to_email}: {str(e)}")
            return False
