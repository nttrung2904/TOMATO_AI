"""Test export PDF function directly"""
import sys
import os
from pathlib import Path

# Add tomato directory to path
sys.path.insert(0, str(Path(__file__).parent / 'tomato'))

# Set BASE_DIR before importing app
os.environ['BASE_DIR'] = str(Path(__file__).parent)

import json
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Import DISEASE_INFO
from app import DISEASE_INFO, BASE_DIR

print("Testing export PDF function...")
print(f"BASE_DIR: {BASE_DIR}")
print()

# Get last prediction
history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
print(f"Reading history from: {history_file}")

if not history_file.exists():
    print("❌ History file not found!")
    sys.exit(1)

prediction = None
with open(history_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            entry = json.loads(line.strip())
            prediction = entry  # Get last one
        except json.JSONDecodeError:
            continue

if not prediction:
    print("❌ No predictions found!")
    sys.exit(1)

prediction_id = prediction['id']
print(f"✓ Found prediction: {prediction_id}")
print(f"  Label: {prediction.get('predicted_label')}")
print(f"  Confidence: {prediction.get('probability', 0) * 100:.1f}%")
print()

# Get disease info
label = prediction.get('predicted_label', '')
disease_info = DISEASE_INFO.get(label, {
    'name': label,
    'definition': 'Không có thông tin',
    'prevention': []
})

print(f"Disease info: {disease_info['name']}")
print()

# Try to create PDF
print("Creating PDF...")
try:
    # Register Vietnamese font
    font_registered = False
    font_path = Path(r'C:\Windows\Fonts\arial.ttf')
    
    if font_path.exists():
        try:
            pdfmetrics.registerFont(TTFont('Vietnamese', str(font_path)))
            font_registered = True
            print("✓ Font registered: Arial")
        except Exception as e:
            print(f"✗ Font registration failed: {e}")
    
    base_font = 'Vietnamese' if font_registered else 'Helvetica'
    base_font_bold = 'Vietnamese' if font_registered else 'Helvetica-Bold'
    
    # Create PDF
    output_path = BASE_DIR / 'test_export_output.pdf'
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                           topMargin=20*mm, bottomMargin=20*mm)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2d5016'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName=base_font_bold
    )
    elements.append(Paragraph('BÁO CÁO DỰ ĐOÁN BỆNH CÀ CHUA', title_style))
    elements.append(Spacer(1, 10*mm))
    
    # Prediction info table
    try:
        ts = datetime.fromisoformat(prediction.get('timestamp', ''))
        formatted_time = ts.strftime('%d/%m/%Y %H:%M:%S')
    except:
        formatted_time = prediction.get('timestamp', 'N/A')
    
    info_data = [
        ['Mã dự đoán:', prediction.get('id', 'N/A')],
        ['Thời gian:', formatted_time],
        ['Model:', prediction.get('model_name', 'N/A')],
        ['Pipeline:', prediction.get('pipeline_key', 'N/A')],
        ['Bệnh phát hiện:', disease_info['name']],
        ['Độ tin cậy:', f"{prediction.get('probability', 0) * 100:.1f}%"],
    ]
    
    info_table = Table(info_data, colWidths=[50*mm, 100*mm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), base_font_bold),
        ('FONTNAME', (1, 0), (1, -1), base_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 10*mm))
    
    # Disease definition
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=base_font_bold,
        fontSize=14,
        textColor=colors.HexColor('#2d5016')
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontName=base_font,
        fontSize=11,
        alignment=TA_LEFT
    )
    
    elements.append(Paragraph('THÔNG TIN BỆNH:', heading_style))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph(disease_info['definition'], body_style))
    elements.append(Spacer(1, 5*mm))
    
    # Prevention measures
    if disease_info.get('prevention'):
        elements.append(Paragraph('BIỆN PHÁP PHÒNG NGỪA:', heading_style))
        elements.append(Spacer(1, 3*mm))
        for i, measure in enumerate(disease_info['prevention'], 1):
            elements.append(Paragraph(f"{i}. {measure}", body_style))
            elements.append(Spacer(1, 2*mm))
    
    # Add image if available
    image_path_str = prediction.get('image_path', '')
    if image_path_str:
        if image_path_str.startswith('/static/'):
            image_path_str = image_path_str[8:]
        
        img_file = BASE_DIR / 'static' / image_path_str
        print(f"Looking for image: {img_file}")
        if img_file.exists():
            print(f"✓ Image found, adding to PDF")
            elements.append(Spacer(1, 5*mm))
            elements.append(Paragraph('ẢNH ĐÃ XỬ LÝ:', heading_style))
            elements.append(Spacer(1, 3*mm))
            try:
                img = RLImage(str(img_file), width=100*mm, height=100*mm)
                elements.append(img)
            except Exception as e:
                print(f"✗ Could not add image: {e}")
                elements.append(Paragraph(f"Không thể thêm ảnh: {str(e)}", body_style))
        else:
            print(f"✗ Image not found: {img_file}")
    
    # Footer
    elements.append(Spacer(1, 10*mm))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER,
        fontName=base_font
    )
    elements.append(Paragraph('Báo cáo được tạo tự động bởi Tomato AI System', footer_style))
    elements.append(Paragraph(f'Ngày xuất: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', footer_style))
    
    # Build PDF
    print("Building PDF...")
    doc.build(elements)
    
    # Save to file
    with open(output_path, 'wb') as f:
        f.write(buffer.getvalue())
    
    print(f"✓ PDF created successfully: {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")
    
except Exception as e:
    print(f"❌ Error creating PDF: {e}")
    import traceback
    traceback.print_exc()
