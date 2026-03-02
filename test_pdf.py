"""Test PDF generation with Vietnamese fonts"""
import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

print("Testing PDF generation with Vietnamese fonts...\n")

# Test 1: Find available fonts
print("=" * 60)
print("Test 1: Searching for fonts on Windows")
print("=" * 60)

font_dir = r'C:\Windows\Fonts'
if Path(font_dir).exists():
    print(f"Font directory exists: {font_dir}")
    font_files = list(Path(font_dir).glob('*.ttf'))
    print(f"Found {len(font_files)} TTF fonts")
    
    # Look for common fonts
    common_fonts = ['arial.ttf', 'Arial.ttf', 'arialuni.ttf', 'times.ttf']
    found_fonts = []
    for font_name in common_fonts:
        font_path = Path(font_dir) / font_name
        if font_path.exists():
            print(f"  ✓ Found: {font_name}")
            found_fonts.append((font_name, font_path))
        else:
            print(f"  ✗ Not found: {font_name}")
else:
    print(f"Font directory not found: {font_dir}")
    found_fonts = []

print()

# Test 2: Register a font
print("=" * 60)
print("Test 2: Registering font")
print("=" * 60)

font_registered = False
if found_fonts:
    font_name, font_path = found_fonts[0]
    try:
        pdfmetrics.registerFont(TTFont('Vietnamese', str(font_path)))
        print(f"✓ Successfully registered: {font_name}")
        font_registered = True
    except Exception as e:
        print(f"✗ Failed to register {font_name}: {e}")
else:
    print("✗ No fonts available to register")

print()

# Test 3: Create a simple PDF
print("=" * 60)
print("Test 3: Creating test PDF")
print("=" * 60)

try:
    output_path = Path(__file__).parent / 'test_output.pdf'
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    if font_registered:
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        
        custom_style = ParagraphStyle(
            'Vietnamese',
            parent=styles['Normal'],
            fontName='Vietnamese',
            fontSize=14,
            alignment=TA_CENTER
        )
        
        elements.append(Paragraph('BÁO CÁO DỰ ĐOÁN BỆNH CÀ CHUA', custom_style))
        elements.append(Paragraph('Test tiếng Việt có dấu: áàảãạ êéè', custom_style))
    else:
        # Fallback without Vietnamese font
        elements.append(Paragraph('Test PDF without Vietnamese font', styles['Normal']))
    
    doc.build(elements)
    print(f"✓ PDF created successfully: {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")
except Exception as e:
    print(f"✗ Failed to create PDF: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Test completed!")
print("=" * 60)
