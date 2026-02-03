"""Script kiểm tra các model Gemini có sẵn"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY không được thiết lập trong file .env")
    exit(1)

print(f"✓ API Key found: {GEMINI_API_KEY[:20]}...")
genai.configure(api_key=GEMINI_API_KEY)

print("\n📋 Danh sách các model Gemini có sẵn:\n")

try:
    models = genai.list_models()
    for model in models:
        # Chỉ hiển thị models hỗ trợ generateContent
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
            print(f"  Display name: {model.display_name}")
            print(f"  Description: {model.description}")
            print()
except Exception as e:
    print(f"❌ Lỗi khi lấy danh sách model: {e}")
    print("\nThử test trực tiếp các model phổ biến:")
    
    test_models = [
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-pro',
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro',
        'models/gemini-pro',
    ]
    
    for model_name in test_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hi")
            print(f"✓ {model_name} - HOẠT ĐỘNG")
        except Exception as e:
            print(f"✗ {model_name} - Lỗi: {str(e)[:80]}")
