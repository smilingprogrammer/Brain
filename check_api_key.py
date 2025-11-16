from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv('GEMINI_API_KEY', '')
valid = key and key != 'your_api_key_here'
print(f"API key configured: {valid}")
print(f"Key starts with: {key[:10] if len(key) > 10 else 'N/A'}")
