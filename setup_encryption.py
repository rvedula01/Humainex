import os
from dotenv import load_dotenv
from crypto_utils import setup_encryption, encrypt_api_key

def main():
    # Load environment variables
    load_dotenv()
    
    # Get the API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return
    
    # Set up encryption and get the key
    key = setup_encryption()
    
    if not key:
        print("Failed to set up encryption. Please check the error messages above.")
        return
    
    # Encrypt the API key
    encrypted_key = encrypt_api_key(api_key, key)
    
    # Update the .env file with the encrypted key
    with open('.env', 'w') as f:
        f.write(f'OPENAI_API_KEY="{encrypted_key}"\n')
    
    print("\nAPI key has been encrypted and saved to .env file.")
    print("Original key has been replaced with the encrypted version.")
    print("\nIMPORTANT: Keep your .encryption_key file secure and never commit it to version control!")
    print("Add .encryption_key to your .gitignore file if you haven't already.")

if __name__ == "__main__":
    main()
