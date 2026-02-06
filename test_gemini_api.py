"""Test Gemini API connection"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
print(f"API Key length: {len(GEMINI_API_KEY)}")
print(f"API Key (first 10 chars): {GEMINI_API_KEY[:10]}...")

if not GEMINI_API_KEY:
    print("ERROR: No API key found!")
    exit(1)

try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✓ API configured successfully")
    
    # Test different models
    model_names = [
        'models/gemini-2.5-flash',
        'models/gemini-flash-latest',
        'models/gemini-2.0-flash',
        'models/gemini-2.5-pro',
    ]
    
    for model_name in model_names:
        print(f"\n--- Testing {model_name} ---")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                "Hello, respond with just 'OK'",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.1,
                )
            )
            
            if response and response.text:
                print(f"✓ {model_name}: SUCCESS")
                print(f"  Response: {response.text[:50]}")
            else:
                print(f"✗ {model_name}: Empty response")
                
        except Exception as e:
            print(f"✗ {model_name}: FAILED")
            print(f"  Error: {str(e)[:150]}")
            
except Exception as e:
    print(f"ERROR: Failed to configure API: {e}")
