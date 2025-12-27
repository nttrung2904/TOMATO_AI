# Tomato Disease Detection Web Application

Web application để phát hiện bệnh trên lá cà chua sử dụng Deep Learning.

## Features

- 🔍 Phát hiện 3 loại bệnh cà chua + healthy
- 🤖 Hỗ trợ 8 model architectures (VGG19, ResNet50, MobileNetV2, v.v.)
- 🎨 5 preprocessing pipelines khác nhau
- 💬 Chatbot hỗ trợ câu hỏi về bệnh cà chua
- 📊 Admin dashboard quản lý feedback
- 🚀 LRU cache tự động quản lý memory

## Setup

### 1. Clone repository

```bash
git clone <repo-url>
cd web_tomato_1
```

### 2. Tạo virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình environment variables

```bash
copy .env.example .env
# Edit .env với thông tin của bạn
```

### 5. Chuẩn bị data

Đảm bảo các folder sau tồn tại:
- `data/` - Chat dataset và sample features
- `model/` - Pre-trained models
- `static/images/tomato_samples/` - Positive samples
- `static/images/not_tomato_samples/` - Negative samples

### 6. Build sample features

```bash
python build_sample_features.py
```

### 7. Run application

```bash
python app.py
```

App sẽ chạy tại: http://localhost:5000

## Project Structure

```
web_tomato_1/
├── app.py                      # Main Flask application
├── utils.py                    # Utility functions
├── build_sample_features.py    # Pre-compute sample features
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── data/                      # Datasets
│   ├── tomato_answer_question.xlsx
│   ├── chat_logs.jsonl
│   └── sample_features.pkl
├── model/                     # Pre-trained models
│   ├── avg_hsv/
│   ├── median_cmyk/
│   └── ...
├── static/                    # Static assets
│   ├── css/
│   ├── images/
│   ├── uploaded/
│   └── feedback/
├── templates/                 # HTML templates
│   ├── index.html
│   ├── result.html
│   ├── chat.html
│   └── ...
├── tools/                     # Utility scripts
└── logs/                      # Application logs
```

## API Endpoints

### Public
- `GET /` - Home page
- `POST /predict` - Dự đoán bệnh từ ảnh
- `GET /chat` - Chatbot interface
- `POST /api/chat` - Chatbot API
- `POST /feedback` - Submit feedback

### Admin
- `GET /admin/feedback` - Xem feedback images
- `POST /admin/feedback_action` - Xử lý feedback
- `GET /api/cache_stats` - Xem cache statistics
- `POST /api/clear_cache` - Clear model cache

## Configuration

Environment variables (xem `.env.example`):
- `SECRET_KEY` - Flask secret key
- `MAX_LOADED_MODELS` - Số model tối đa trong cache
- `MIN_MODEL_CONF` - Threshold confidence
- `LOG_LEVEL` - Logging level

## Troubleshooting

### Import Error: No module named 'PIL'
```bash
pip install Pillow
```

### Model not loading
Check logs tại `logs/app.log` và `logs/error.log`

### Out of Memory
Giảm `MAX_LOADED_MODELS` trong `.env`

## License

MIT License

---

## 🔍 Cơ chế Similarity Check

### Tổng quan

Hệ thống sử dụng **bộ lọc thông minh** để phát hiện ảnh "không phải lá cà chua" **TRƯỚC KHI** đưa vào các model dự đoán bệnh.

### Kiến trúc

```
Ảnh đầu vào
    ↓
┌─────────────────────────────────────────────┐
│ SIMILARITY CHECK (Pre-filtering)            │
│ • Sử dụng MobileNetV2 làm feature extractor │
│ • Tính 2 loại features:                     │
│   1. Histogram (màu sắc)                    │
│   2. Deep embeddings (MobileNetV2)          │
│ • So sánh với positive & negative samples   │
└─────────────────────────────────────────────┘
    ↓
  Quyết định: OK / WARNING / REJECTED
    ↓
┌─────────────────────────────────────────────┐
│ MODEL PREDICTION                            │
│ Chọn 1 trong 8 models + 5 pipelines        │
│ Output: 4 classes (3 bệnh + healthy)        │
└─────────────────────────────────────────────┘
```

### Vai trò của MobileNetV2

**TẤT CẢ MODELS** đều được bảo vệ bởi **cùng 1 bộ lọc similarity check** sử dụng MobileNetV2:

| Model dự đoán | Có similarity check? |
|---------------|---------------------|
| VGG19, VGG16, ResNet50 | ✅ CÓ (MobileNetV2) |
| MobileNetV2 | ✅ CÓ (MobileNetV2) |
| InceptionV3, DenseNet, Xception, CNN | ✅ CÓ (MobileNetV2) |

### Tại sao dùng MobileNetV2 cho similarity?

- ✅ **Nhẹ & Nhanh** - Chỉ 3.5M params, ~200ms
- ✅ **Pre-trained ImageNet** - Biết 1000 classes
- ✅ **Embeddings chất lượng cao** - Phân biệt tốt objects
- ✅ **Không cần retrain** - Dùng ngay weights có sẵn

### Cơ chế tính toán

```python
# 1. Trích xuất features
hist = cv2.calcHist(img, [0,1,2], None, [8,8,8], [0,256]*3)
embedding = mobilenet_v2.predict(img)  # (1280,) normalized

# 2. So sánh với samples
pos_sim = 0.5 * hist_sim + 0.5 * deep_sim  # positive samples
neg_sim = 0.5 * hist_sim + 0.5 * deep_sim  # negative samples

# 3. Quyết định
if neg_sim >= 0.75 and pos_sim < 0.60:
    return "REJECTED - Not tomato"
elif pos_sim < 0.40 and confidence < 0.85:
    return "WARNING - Possibly not tomato"
else:
    return "OK - Continue prediction"
```

### Ví dụ thực tế

**Ảnh con chó 🐕:**
- pos_sim = 0.35 (< 0.60) → Không giống lá cà chua
- neg_sim = 0.87 (≥ 0.75) → Giống "không phải lá"
- Kết quả: **REJECTED** ❌

**Lá cà chua khỏe 🍃:**
- pos_sim = 0.92 (≥ 0.60) → Giống lá cà chua
- neg_sim = 0.15 (< 0.75) → Không giống "không phải lá"
- Kết quả: **APPROVED** ✅ → Dự đoán: "Tomato_healthy" (95%)

### Cấu hình ngưỡng

Trong file `.env`:
```env
POS_SIM_THRESH=0.60  # Ngưỡng tối thiểu cho lá cà chua
NEG_SIM_THRESH=0.75  # Ngưỡng từ chối
```

**Điều chỉnh:**
- Hệ thống từ chối quá nhiều lá thật → Giảm `POS_SIM_THRESH=0.50`
- Hệ thống chấp nhận nhiều ảnh sai → Tăng `POS_SIM_THRESH=0.70`

### FAQ

**Q: Tại sao không train models với class "Not_Tomato"?**  
A: "Not tomato" là tập vô hạn (động vật, người, đồ vật...). Similarity check nhanh, chính xác, không cần retrain.

**Q: Làm sao cải thiện độ chính xác?**  
A: 
1. Thêm nhiều ảnh vào `tomato_samples/` và `not_tomato_samples/`
2. Click "Rebuild Sample Features" trong admin panel
3. Điều chỉnh ngưỡng trong `.env`
