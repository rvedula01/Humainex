import os
from dotenv import load_dotenv

def test_env():
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if SENDGRID_API_KEY is set
    sendgrid_key = os.getenv('SENDGRID_API_KEY')
    print(f"SENDGRID_API_KEY is set: {bool(sendgrid_key)}")
    print(f"Key length: {len(sendgrid_key) if sendgrid_key else 0} characters")
    
    # Check if we can initialize SendGrid client
    try:
        from sendgrid import SendGridAPIClient
        if sendgrid_key:
            sg = SendGridAPIClient(api_key=sendgrid_key)
            print("Successfully initialized SendGrid client!")
        else:
            print("No SENDGRID_API_KEY found in environment variables")
    except Exception as e:
        print(f"Error initializing SendGrid client: {e}")

if __name__ == "__main__":
    test_env()
