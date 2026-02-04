import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv

def generate_key():
    """Generate a new Fernet key."""
    return Fernet.generate_key().decode()

def save_key(key, key_file='.encryption_key'):
    """Save the encryption key to a file."""
    with open(key_file, 'w') as f:
        f.write(key)
    print(f"Encryption key saved to {key_file}. Keep this file secure and don't commit it to version control!")

def load_key(key_file='.encryption_key'):
    """Load the encryption key from a file."""
    try:
        with open(key_file, 'r') as f:
            return f.read().encode()
    except FileNotFoundError:
        print(f"Error: Key file '{key_file}' not found.")
        return None

def encrypt_api_key(api_key, key):
    """Encrypt an API key using Fernet."""
    f = Fernet(key)
    return f.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key, key):
    """Decrypt an API key using Fernet."""
    f = Fernet(key)
    return f.decrypt(encrypted_key.encode()).decode()

def setup_encryption():
    """Set up encryption by generating a new key if one doesn't exist."""
    key_file = '.encryption_key'
    if not os.path.exists(key_file):
        key = generate_key()
        save_key(key, key_file)
        print(f"A new encryption key has been generated and saved to {key_file}")
        return key
    return load_key(key_file)

# Example usage:
if __name__ == "__main__":
    # Generate and save a new key if it doesn't exist
    key = setup_encryption()
    
    # Example: Encrypt an API key
    # api_key = "your_actual_api_key_here"
    # encrypted = encrypt_api_key(api_key, key)
    # print(f"Encrypted API key: {encrypted}")
    
    # Example: Decrypt an API key
    # decrypted = decrypt_api_key(encrypted, key)
    # print(f"Decrypted API key: {decrypted}")
