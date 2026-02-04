import os
from dotenv import load_dotenv

def check_environment():
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if OPENAI_API_KEY exists
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("Environment Variables Check")
    print("=========================")
    print(f"Current working directory: {os.getcwd()}")
    print(f"OPENAI_API_KEY exists: {'Yes' if api_key else 'No'}")
    
    if api_key:
        print(f"API Key starts with: {api_key[:5]}...{api_key[-4:] if len(api_key) > 9 else ''}")
        print(f"API Key length: {len(api_key)} characters")
    
    # List all environment variables (be careful with this in shared environments)
    print("\nAll environment variables:")
    for key, value in os.environ.items():
        if "KEY" in key or "SECRET" in key or "PASS" in key or "TOKEN" in key:
            print(f"{key}: {'*' * 8}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    check_environment()
