import os
from dotenv import load_dotenv
from crypto_utils import load_key, decrypt_api_key

def get_api_key():
    """Load and decrypt the API key from environment variables."""
    # Load environment variables
    load_dotenv()
    
    # Get the encrypted API key
    encrypted_key = os.getenv('OPENAI_API_KEY')
    if not encrypted_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Load the encryption key
    key = load_key()
    if not key:
        raise ValueError("Failed to load encryption key. Make sure .encryption_key file exists.")
    
    # Decrypt and return the API key
    try:
        return decrypt_api_key(encrypted_key, key)
    except Exception as e:
        raise ValueError(f"Failed to decrypt API key: {str(e)}")

# Example usage in other files:
# from config import get_api_key
# api_key = get_api_key()
