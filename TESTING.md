# Hướng dẫn Testing - Tomato Disease Detection

## 📋 Mục lục
1. [Kiểm tra cài đặt](#kiểm-tra-cài-đặt)
2. [Test chức năng chính](#test-chức-năng-chính)
3. [Test cases chi tiết](#test-cases-chi-tiết)
4. [Test performance](#test-performance)
5. [Test security](#test-security)

---

## Kiểm tra cài đặt

### ✅ Kiểm tra Python dependencies
```bash
cd tomato
python -c "import flask; import tensorflow; import cv2; import pandas; print('All imports OK')"
```

**Kết quả mong đợi:**
```
All imports OK
```

### ✅ Kiểm tra cấu trúc thư mục
```bash
ls -la ../data/
ls -la ../model/average_hsv/
ls -la ../static/images/tomato_samples/
```

**Kết quả mong đợi:**
- `data/`: Có `sample_features.pkl`, `tomato_answer_question.xlsx`
- `model/average_hsv/`: Có ít nhất 1 file `.keras`
- `static/images/tomato_samples/`: Có ít nhất 50 ảnh

### ✅ Kiểm tra syntax Python
```bash
python -m py_compile app.py utils.py build_sample_features.py
echo "Syntax check: OK"
```

---

## Test chức năng chính

### 1️⃣ Test khởi động server

**Bước thực hiện:**
```bash
python app.py
```

**Kết quả mong đợi:**
```
[INFO] ==================== PRELOAD PHASE ====================
[INFO] Discovering models in d:\...\model
[INFO] Found 40 model configurations
[INFO] Preloading default model: VGG19_average_hsv
[INFO] Model loaded successfully from cache
[INFO] Loaded chat dataset: 605 questions
[INFO] Loaded sample features: 352 positive, 104 negative
[INFO] Application preload completed successfully
[INFO] Starting Flask development server...
 * Running on http://0.0.0.0:5000
```

**Checklist:**
- [ ] Server khởi động không lỗi
- [ ] Phát hiện đúng số models (40 configs)
- [ ] Load được VGG19 mặc định
- [ ] Load được chat dataset (605 questions)
- [ ] Load được sample features (352 pos, 104 neg)
- [ ] Browser tự mở (nếu `AUTO_OPEN_BROWSER=true`)

---

### 2️⃣ Test Homepage (GET /)

**Bước thực hiện:**
1. Mở browser: http://localhost:5000
2. Kiểm tra giao diện

**Checklist:**
- [ ] Trang hiển thị không lỗi 500
- [ ] Tiêu đề: "Nhận diện bệnh cà chua"
- [ ] Có dropdown chọn model (8 options)
- [ ] Có dropdown chọn pipeline (5 options)
- [ ] Có nút "Chọn ảnh" upload
- [ ] Menu navigation có: Trang chủ, Giới thiệu, Hỏi đáp, Quản lý

**Screenshot:** Chụp màn hình lưu vào báo cáo

---

### 3️⃣ Test Prediction (POST /predict)

#### Test Case 3.1: Upload ảnh lá cà chua khỏe mạnh

**Dữ liệu test:**
- File: `static/images/tomato_samples/` (chọn bất kỳ)
- Model: VGG19
- Pipeline: average_hsv

**Bước thực hiện:**
1. Chọn file ảnh lá cà chua
2. Chọn model: VGG19
3. Chọn pipeline: average_hsv
4. Click "Dự đoán"

**Kết quả mong đợi:**
- [ ] Loading spinner hiển thị
- [ ] Redirect đến `/result`
- [ ] Hiển thị ảnh đã upload
- [ ] Hiển thị kết quả: "Tomato_healthy" (hoặc một trong 3 bệnh)
- [ ] Hiển thị confidence % (> 85%)
- [ ] Hiển thị thông tin bệnh (định nghĩa + biện pháp phòng ngừa)
- [ ] KHÔNG hiển thị warning "possibly not tomato"
- [ ] KHÔNG hiển thị nút feedback (nếu confidence cao)

**Log server:**
```
[INFO] ========================================
[INFO] New prediction request [ID: abc123]
[INFO] Step 1: Validating request parameters
[INFO] Step 2: Validating and decoding image
[INFO] Step 3: Preparing image for prediction
[INFO] Step 4: Running model prediction
[INFO] Model loaded successfully from cache
[INFO] Prediction completed in 0.234 seconds
[INFO] Step 5: Processing prediction results
[INFO] Predicted: Tomato_healthy (confidence: 95.67%)
[INFO] Step 8: Assessing prediction quality
[INFO] Sample check: pos_sim=0.852, neg_sim=0.123, combined=0.701
```

#### Test Case 3.2: Upload ảnh KHÔNG phải lá cà chua

**Dữ liệu test:**
- File: `static/images/not_tomato_samples/` (chọn bất kỳ)
- Model: VGG19
- Pipeline: average_hsv

**Kết quả mong đợi:**
- [ ] Hiển thị warning: "⚠️ Ảnh này có thể không phải lá cà chua"
- [ ] Vẫn hiển thị kết quả dự đoán
- [ ] Hiển thị nút feedback: "✅ Đúng là lá cà chua" và "❌ Không phải lá cà chua"
- [ ] Similarity info: `pos_sim < 0.40` hoặc `neg_sim >= 0.75`

#### Test Case 3.3: Upload ảnh bệnh Early Blight

**Dữ liệu test:**
- File: Ảnh lá cà chua bị bệnh sớm
- Model: VGG19
- Pipeline: average_hsv

**Kết quả mong đợi:**
- [ ] Predicted label: "Tomato_Early_blight"
- [ ] Confidence: > 85%
- [ ] Hiển thị định nghĩa bệnh (do nấm Alternaria solani)
- [ ] Hiển thị 5 biện pháp phòng ngừa
- [ ] Không có warning nếu similarity scores tốt

#### Test Case 3.4: Upload file không hợp lệ

**Dữ liệu test:**
- File: `.txt`, `.pdf`, `.zip`

**Kết quả mong đợi:**
- [ ] Flash message: "Định dạng file không hợp lệ"
- [ ] Không redirect, ở lại homepage
- [ ] Log: `[WARNING] Invalid file extension: test.txt`

#### Test Case 3.5: Không chọn file

**Kết quả mong đợi:**
- [ ] Flash message: "Bạn chưa chọn file."
- [ ] Ở lại homepage

---

### 4️⃣ Test Chatbot (POST /api/chat)

#### Test Case 4.1: Câu hỏi về bệnh cà chua

**Bước thực hiện:**
1. Truy cập: http://localhost:5000/chat
2. Nhập câu hỏi: "Bệnh sớm là gì?"
3. Click "Gửi"

**Kết quả mong đợi:**
- [ ] Loading animation hiển thị
- [ ] Response hiển thị trong 2 giây
- [ ] Câu trả lời liên quan đến bệnh Early blight
- [ ] Có avatar bot
- [ ] Format text đúng (không có HTML entities)

**Kiểm tra fuzzy matching:**
```
Input: "benh som la gi"        → Kết quả: Trả lời về Early blight
Input: "cach phong ngua"       → Kết quả: Biện pháp phòng ngừa chung
Input: "tom"                   → Kết quả: "Không tìm thấy câu trả lời"
```

#### Test Case 4.2: Câu hỏi random không liên quan

**Input:** "Trời hôm nay đẹp quá"

**Kết quả mong đợi:**
- [ ] Response: "Xin lỗi, tôi không tìm thấy câu trả lời phù hợp..."
- [ ] Gợi ý các câu hỏi mẫu

---

### 5️⃣ Test Feedback System

#### Test Case 5.1: User confirm feedback

**Bước thực hiện:**
1. Upload ảnh có warning "possibly not tomato"
2. Click nút "✅ Đúng là lá cà chua"

**Kết quả mong đợi:**
- [ ] Success message: "Đã lưu feedback. Quản trị viên sẽ xem xét."
- [ ] File được lưu vào: `static/feedback/confirmed_tomato/YYYYMMDD_HHMMSS_confirmed_tomato.png`
- [ ] KHÔNG tự động rebuild `sample_features.pkl`
- [ ] Cần admin xử lý manually

#### Test Case 5.2: User reject feedback

**Bước thực hiện:**
1. Upload ảnh tomato
2. Click nút "❌ Không phải lá cà chua"

**Kết quả mong đợi:**
- [ ] Success message hiển thị
- [ ] File được lưu vào: `static/feedback/not_tomato/YYYYMMDD_HHMMSS_not_tomato.png`

---

### 6️⃣ Test Admin Panel

#### Test Case 6.1: Truy cập admin panel

**Bước thực hiện:**
1. Truy cập: http://localhost:5000/admin/feedback
2. Nhập username/password từ `.env`

**Kết quả mong đợi:**
- [ ] Hiển thị HTTP Basic Auth dialog
- [ ] Nhập sai → 401 Unauthorized
- [ ] Nhập đúng → Hiển thị feedback images

#### Test Case 6.2: Admin thêm feedback vào samples

**Bước thực hiện:**
1. Login admin panel
2. Chọn 1 ảnh trong `confirmed_tomato`
3. Click "Thêm vào mẫu dương tính"
4. Chờ thông báo

**Kết quả mong đợi:**
- [ ] Success toast: "Đã xử lý 1 ảnh"
- [ ] File được di chuyển từ `feedback/confirmed_tomato/` → `images/tomato_samples/`
- [ ] Background thread tự động rebuild `sample_features.pkl`
- [ ] Log: `[INFO] Starting background rebuild of sample features`
- [ ] Log: `[INFO] Background rebuild completed`

#### Test Case 6.3: Admin rebuild manual

**Bước thực hiện:**
1. Click nút "🔄 Rebuild Sample Features"
2. Đợi 10-30 giây

**Kết quả mong đợi:**
- [ ] Toast: "Đã bắt đầu rebuild (chạy nền)"
- [ ] Log: `[INFO] Starting background rebuild...`
- [ ] File `data/sample_features.pkl` được cập nhật (check timestamp)

#### Test Case 6.4: Admin reload cache

**Bước thực hiện:**
1. Click nút "♻️ Reload Sample Cache"

**Kết quả mong đợi:**
- [ ] Toast: "Reloaded sample features (positive=352, negative=104)"
- [ ] Số lượng cập nhật nếu có rebuild trước đó

#### Test Case 6.5: Export chat logs

**Bước thực hiện:**
1. Click nút "📥 Export Chat Logs"

**Kết quả mong đợi:**
- [ ] File CSV download: `chat_logs_YYYYMMDD_HHMMSS.csv`
- [ ] Mở được bằng Excel
- [ ] UTF-8 encoding đúng (tiếng Việt không bị lỗi font)
- [ ] Có 3 cột: ts, question, answer

---

## Test Performance

### 📊 Test 1: Prediction latency

**Công cụ:** Stopwatch hoặc browser DevTools

**Bước thực hiện:**
1. Upload ảnh lần 1 (cold start)
2. Đo thời gian từ click "Dự đoán" → Hiển thị kết quả
3. Upload cùng ảnh lần 2 (cache hit)
4. Đo thời gian lần 2

**Kết quả mong đợi:**
- Lần 1 (cold): 2-5 giây
- Lần 2 (cache): 0.5-2 giây
- Log: `[INFO] Prediction completed in 0.234 seconds`

### 📊 Test 2: Memory usage

**Công cụ:** Task Manager (Windows) hoặc `htop` (Linux)

**Bước thực hiện:**
1. Khởi động server, note RAM usage ban đầu
2. Upload 10 ảnh với 2 models khác nhau
3. Kiểm tra RAM usage

**Kết quả mong đợi:**
- RAM tăng khi load model mới (1-2 GB/model)
- RAM ổn định sau khi đạt `MAX_LOADED_MODELS`
- Không có memory leak (RAM không tăng vô hạn)

### 📊 Test 3: Cache statistics

**Bước thực hiện:**
```bash
curl http://localhost:5000/api/cache_stats
```

**Kết quả mong đợi:**
```json
{
  "cache_size": 2,
  "hit_rate": 66.67,
  "hits": 4,
  "misses": 2,
  "evictions": 0,
  "keys": ["VGG19_average_hsv", "ResNet50_median_cmyk"]
}
```

**Checklist:**
- [ ] `hit_rate` tăng dần khi test nhiều lần
- [ ] `cache_size` không vượt quá `MAX_LOADED_MODELS`
- [ ] `evictions` tăng khi load > MAX_LOADED_MODELS models

---

## Test Security

### 🔒 Test 1: Admin authentication

**Test Case:** Truy cập admin endpoints không có auth

```bash
curl -X GET http://localhost:5000/admin/feedback
```

**Kết quả mong đợi:**
```
401 Unauthorized
WWW-Authenticate: Basic realm="Authentication Required"
```

### 🔒 Test 2: Path traversal protection

**Test Case:** Admin action với malicious path

**Request:**
```json
POST /admin/feedback_action
{
  "action": "add_to_samples",
  "items": [
    {"dir": "../../../etc", "name": "passwd"}
  ]
}
```

**Kết quả mong đợi:**
- [ ] Log: `[WARNING] Skipping invalid dir name: ../../../etc`
- [ ] Response: `{"ok": true, "message": "Đã xử lý 0 ảnh"}`
- [ ] File `/etc/passwd` KHÔNG bị di chuyển

### 🔒 Test 3: File upload limits

**Test Case:** Upload file > 16MB

**Kết quả mong đợi:**
- [ ] Error: "Request Entity Too Large"
- [ ] Server không crash

### 🔒 Test 4: SQL Injection (N/A)

Ứng dụng không dùng SQL database → Không có SQL injection risk.

---

## Test Cross-browser Compatibility

### 🌐 Browsers to test
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Edge (latest)
- [ ] Safari (macOS/iOS)

**Checklist mỗi browser:**
- [ ] Homepage render đúng
- [ ] Upload file hoạt động
- [ ] Prediction hiển thị kết quả
- [ ] Chatbot gửi/nhận message
- [ ] Admin panel login và chức năng

---

## Test Edge Cases

### ⚠️ Edge Case 1: Empty file upload

**Bước:** Upload file 0 byte

**Kết quả mong đợi:**
- [ ] Error: "File rỗng, vui lòng chọn file khác"

### ⚠️ Edge Case 2: Corrupted image

**Bước:** Upload file `.jpg` bị lỗi

**Kết quả mong đợi:**
- [ ] Error: "Ảnh hỏng hoặc không thể xác thực"

### ⚠️ Edge Case 3: Ảnh quá lớn (> 3000x3000)

**Kết quả mong đợi:**
- [ ] Server tự động resize xuống 3000px
- [ ] Log: "Image resized from ..."
- [ ] Prediction vẫn hoạt động

### ⚠️ Edge Case 4: Model file bị xóa

**Bước:**
1. Xóa file model đang được cache
2. Clear cache: `POST /api/clear_cache`
3. Upload ảnh với model đó

**Kết quả mong đợi:**
- [ ] Error: "Model not found"
- [ ] Server không crash

---

## Automated Testing (Optional)

### Unit Tests với pytest

Tạo file `test_app.py`:
```python
import pytest
from app import app, validate_request_parameters

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Tomato' in rv.data

def test_predict_no_file(client):
    rv = client.post('/predict', data={})
    assert b'file' in rv.data.lower()

def test_admin_no_auth(client):
    rv = client.get('/admin/feedback')
    assert rv.status_code == 401
```

Chạy tests:
```bash
pip install pytest
pytest test_app.py -v
```

---

## Regression Testing

Sau mỗi lần sửa code, chạy lại:

1. ✅ Python syntax check
2. ✅ Server khởi động không lỗi
3. ✅ Upload ảnh tomato → Prediction OK
4. ✅ Upload ảnh non-tomato → Warning hiển thị
5. ✅ Chatbot response câu hỏi
6. ✅ Admin add feedback → Rebuild successful
7. ✅ Cache statistics correct

---

## Báo cáo Test cho Khóa luận

### Template báo cáo

```
PHẦN PHỤ LỤC - KẾT QUẢ KIỂM THỬ HỆ THỐNG

1. MÔI TRƯỜNG KIỂM THỬ
   - Hệ điều hành: Windows 11 / Ubuntu 22.04
   - Python: 3.10.12
   - RAM: 16GB
   - Browser: Chrome 120.0

2. KẾT QUẢ KIỂM THỬ CHỨC NĂNG
   ┌─────────────────────────────┬─────────┬──────────┐
   │ Chức năng                   │ Kết quả │ Ghi chú  │
   ├─────────────────────────────┼─────────┼──────────┤
   │ Upload ảnh & dự đoán        │ PASS    │ 10/10    │
   │ Chatbot hỏi đáp             │ PASS    │ 20/20    │
   │ Feedback system             │ PASS    │ 5/5      │
   │ Admin panel quản lý         │ PASS    │ 8/8      │
   │ Cache & performance         │ PASS    │ OK       │
   │ Security & authentication   │ PASS    │ OK       │
   └─────────────────────────────┴─────────┴──────────┘

3. HIỆU NĂNG
   - Prediction latency (cold): 3.2s
   - Prediction latency (cache): 0.8s
   - Memory usage: Stable (~4GB with 2 models)

4. KẾT LUẬN
   Hệ thống hoạt động ổn định, đáp ứng yêu cầu nghiệp vụ.
```

### Screenshots cần chụp

1. Homepage với dropdown models/pipelines
2. Kết quả prediction thành công (Tomato_healthy)
3. Warning "possibly not tomato" với nút feedback
4. Chatbot conversation (3-4 câu hỏi)
5. Admin panel hiển thị feedback images
6. Cache statistics API response
7. Server logs khi prediction thành công

---

**Chúc bạn testing thành công!** ✅
