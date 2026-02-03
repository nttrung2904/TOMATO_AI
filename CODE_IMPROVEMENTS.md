# 📋 Đề xuất cải thiện code

## 🗑️ 1. CODE DƯ THỪA CẦN XÓA

### A. Comment code không dùng (app.py)
**Vị trí**: Dòng 1260-1263
```python
# elif pipeline_key in ['bilateral_lab', 'average_lab']:
#     # Chuyển từ LAB sang BGR, rồi BGR sang RGB
#     out_bgr = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
#     out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
```
**Lý do xóa**: Pipeline LAB không còn được sử dụng, comment này gây rối

### B. Reference không tồn tại
**Vị trí**: Dòng 1256
```python
elif pipeline_key in ['bilateral_hsv', 'average_hsv']:
```
**Sửa thành**: 
```python
elif pipeline_key in ['average_hsv']:
```
**Lý do**: `bilateral_hsv` không có trong PIPELINES dict

---

## ⚡ 2. CẢI THIỆN PERFORMANCE

### A. Tách function dài
**Function**: `preprocess_image_for_model()` (100+ dòng)

**Đề xuất**: Tách thành các sub-functions:
```python
def _apply_pipeline(img_rgb, pipeline_key):
    """Apply preprocessing pipeline"""
    # ... logic hiện tại

def _normalize_channels(img):
    """Ensure image has 3 channels"""
    # ... logic normalize channels

def _resize_to_target(img, target_size):
    """Resize image to target size"""
    # ... logic resize

def preprocess_image_for_model(image_bgr, pipeline_key):
    """Main preprocessing function"""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
    
    processed = _apply_pipeline(img_resized, pipeline_key)
    processed = _normalize_channels(processed)
    processed = _resize_to_target(processed, IMG_SIZE)
    
    return np.expand_dims(processed, axis=0)
```

### B. Gemini API - Retry logic
**Vị trí**: `get_gemini_response()` dòng 736-801

**Thêm**: Exponential backoff cho rate limit
```python
import time

def get_gemini_response(user_question: str, max_retries=3) -> str:
    for attempt in range(max_retries):
        try:
            response = GEMINI_MODEL.generate_content(...)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 1  # 1s, 2s, 4s
                app.logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise
```

### C. Cache warming - Preload thêm models
**Vị trí**: `preload()` function dòng 2385-2420

**Đề xuất**: Preload top 3 models thường dùng
```python
def preload():
    # ... existing code ...
    
    # Preload popular models
    popular_models = [
        ('VGG19', 'average_hsv'),
        ('Xception', 'gb_noise_cmyk'),
        ('MobileNetV2', 'median_hsi')
    ]
    
    for model_name, pipeline in popular_models[:2]:  # Load 2 models
        try:
            load_model_by_name(model_name, pipeline)
            app.logger.info(f"Preloaded {model_name} + {pipeline}")
        except Exception as e:
            app.logger.warning(f"Failed to preload {model_name}: {e}")
```

---

## 🔒 3. SECURITY & VALIDATION

### A. File upload MIME type validation
**Vị trí**: `validate_and_decode_image()` dòng 1794-1868

**Thêm check MIME type**:
```python
import magic  # pip install python-magic-bin

def validate_and_decode_image(file_obj):
    raw_bytes = file_obj.read()
    
    # Check MIME type
    mime = magic.from_buffer(raw_bytes, mime=True)
    if mime not in ['image/jpeg', 'image/png', 'image/jpg']:
        raise ValidationError(
            f"Invalid MIME type: {mime}",
            user_message=f"File không phải ảnh hợp lệ (MIME: {mime})"
        )
    
    # ... rest of code
```

### B. Admin password strength
**Vị trí**: Dòng 470-471

**Đề xuất**: Thêm warning nếu dùng password yếu
```python
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')

# Warn về password yếu
if ADMIN_PASSWORD in ['admin123', 'admin', 'password', '123456']:
    app.logger.error(
        "⚠️  WEAK ADMIN PASSWORD DETECTED! "
        "Please set a strong ADMIN_PASSWORD in .env file"
    )
```

### C. Rate limiting cho API endpoints
**Thêm**: Flask-Limiter để chống spam

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")  # Giới hạn 10 dự đoán/phút
def predict():
    # ... existing code
```

---

## 🎨 4. CODE QUALITY

### A. Extract magic numbers to constants
**Vị trí**: Nhiều nơi trong code

**Tạo file constants**:
```python
# tomato/constants.py

# Similarity thresholds
SIMILARITY_POS_THRESHOLD = 0.60
SIMILARITY_NEG_THRESHOLD = 0.75
SIMILARITY_NEG_STRONG_THRESHOLD = 0.65
SIMILARITY_NEG_WEAK_THRESHOLD = 0.60
SIMILARITY_POS_VERY_LOW = 0.40

# Model confidence thresholds
MODEL_CONF_MIN = 0.85
MODEL_CONF_FEEDBACK_THRESHOLD = 0.95

# Embedding weights
EMBEDDING_WEIGHT_DEEP = 0.5
EMBEDDING_WEIGHT_HIST = 0.5

# Image processing
IMAGE_RESIZE_INTERPOLATION = cv2.INTER_AREA
GREEN_RATIO_THRESHOLD = 0.05
```

### B. Refactor duplicate history reading logic
**Vị trí**: `history()`, `view_prediction()`, `clear_history()`

**Tạo helper function**:
```python
def _read_history_file():
    """Read and parse history file"""
    history_file = BASE_DIR / 'data' / 'prediction_history.jsonl'
    history_list = []
    
    if not history_file.exists():
        return history_list
    
    with open(history_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # Format timestamp
                try:
                    ts = datetime.fromisoformat(entry.get('timestamp', ''))
                    entry['formatted_time'] = ts.strftime('%d/%m/%Y %H:%M:%S')
                except:
                    entry['formatted_time'] = entry.get('timestamp', 'N/A')
                
                # Get disease info
                label = entry.get('predicted_label', '')
                if label in DISEASE_INFO:
                    entry['disease_name'] = DISEASE_INFO[label]['name']
                else:
                    entry['disease_name'] = label
                
                history_list.append(entry)
            except json.JSONDecodeError:
                continue
    
    return history_list

# Sử dụng:
@app.route('/history')
def history():
    try:
        history_list = _read_history_file()
        history_list.reverse()  # Mới nhất trước
        return render_template('history.html', history=history_list)
    except Exception as e:
        app.logger.exception('Error loading history')
        flash('Không thể tải lịch sử dự đoán')
        return redirect(url_for('index'))
```

### C. Type hints cho functions quan trọng
**Thêm type annotations**:
```python
from typing import Dict, List, Tuple, Optional
import numpy.typing as npt

def preprocess_image_for_model(
    image_bgr: npt.NDArray[np.uint8], 
    pipeline_key: str
) -> npt.NDArray[np.float32]:
    """Preprocess image for model input"""
    # ...

def compute_sample_similarity(
    img_bgr: npt.NDArray[np.uint8]
) -> Dict[str, any]:
    """Compute similarity metrics"""
    # ...

def load_model_by_name(
    arch_name: str, 
    pipeline_key: str
) -> Tuple[tf.keras.Model, List[str]]:
    """Load model by architecture name and pipeline key"""
    # ...
```

---

## 📦 5. DEPENDENCIES CẦN THÊM

Thêm vào `requirements.txt`:
```txt
# Security & Rate Limiting
Flask-Limiter==3.5.0

# MIME type detection
python-magic-bin==0.4.14  # Windows
# python-magic==0.4.27     # Linux/Mac

# Type checking (dev dependency)
mypy==1.7.1
numpy-stubs==1.25.0
```

---

## 🧪 6. TESTING - NÊN THÊM

Tạo file `tests/test_app.py`:
```python
import pytest
from tomato.app import app, validate_request_parameters, is_leaf_like
import cv2
import numpy as np

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Test trang chủ load được"""
    response = client.get('/')
    assert response.status_code == 200

def test_is_leaf_like():
    """Test function kiểm tra ảnh lá"""
    # Tạo ảnh giả màu xanh
    green_img = np.zeros((224, 224, 3), dtype=np.uint8)
    green_img[:, :, 1] = 200  # Green channel
    assert is_leaf_like(green_img) == True
    
    # Tạo ảnh không phải lá (đỏ)
    red_img = np.zeros((224, 224, 3), dtype=np.uint8)
    red_img[:, :, 2] = 200  # Red channel
    assert is_leaf_like(red_img) == False

def test_chat_api(client):
    """Test chatbot API"""
    response = client.post('/api/chat', 
                          json={'q': 'Cà chua là gì?'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'answer' in data
```

---

## 🎯 7. IMPROVEMENTS THEO PRIORITY

### HIGH PRIORITY (Làm ngay):
1. ✅ Xóa code comment LAB không dùng
2. ✅ Sửa reference `bilateral_hsv`
3. ✅ Thêm warning password yếu
4. ✅ Refactor duplicate history logic

### MEDIUM PRIORITY (Làm trong tuần):
5. ⚡ Tách function `preprocess_image_for_model`
6. ⚡ Extract magic numbers ra constants
7. ⚡ Thêm retry logic cho Gemini
8. 🔒 Thêm MIME type validation

### LOW PRIORITY (Nice to have):
9. 📦 Thêm Flask-Limiter
10. 🧪 Viết unit tests
11. 📝 Thêm type hints
12. ⚡ Cache thêm models

---

## 📊 CODE METRICS

**Current State:**
- Total lines: 2489
- Functions: ~45
- Average function length: ~55 lines
- Longest function: `predict()` - 120 lines

**Target State:**
- Total lines: ~2200 (giảm 12%)
- Functions: ~55 (tách nhỏ)
- Average function length: ~40 lines
- Longest function: <80 lines

---

## 🚀 NEXT STEPS

1. Backup code hiện tại
2. Implement HIGH priority items
3. Test kỹ sau mỗi thay đổi
4. Commit từng nhóm changes
5. Deploy lên test environment
6. Monitor performance improvements

---

**Lưu ý**: Không thực hiện tất cả cùng lúc. Làm từng nhóm nhỏ, test kỹ, rồi mới chuyển sang nhóm tiếp theo.
