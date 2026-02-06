"""List available Gemini models"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Available Gemini models:\n")
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
            print(f"  Display name: {model.display_name}")
            print(f"  Description: {model.description[:80]}...")
            print()
            
except Exception as e:
    print(f"Error: {e}")
