# Computer Vision Features - Implementation Summary

## 🎯 Các tính năng CV đã được thêm vào hệ thống

### 1. ✅ **Disease Localization với Grad-CAM** 
**Endpoint**: `/api/gradcam`

**Chức năng**: 
- Tạo heatmap highlight vùng bệnh trên lá cà chua
- Sử dụng Gradient-weighted Class Activation Mapping (Grad-CAM)
- Overlay heatmap lên ảnh gốc với màu sắc trực quan

**Cách sử dụng**:
- Truy cập trang kết quả dự đoán
- Click nút "🔥 Xem vùng bệnh (Heatmap)"
- Heatmap sẽ hiển thị trong modal với:
  - 🔴 Vùng đỏ: Nơi model phát hiện triệu chứng bệnh
  - 🔵 Vùng xanh: Vùng khỏe mạnh

**Code location**:
- Backend: `utils.py` - `generate_gradcam()`, `overlay_heatmap_on_image()`
- API: `app.py` - Route `/api/gradcam`
- Frontend: `result.html` - JavaScript modal

---

### 2. ✅ **Real-time Webcam Detection**
**Page**: `/webcam`

**Chức năng**:
- Scan lá cà chua real-time qua webcam
- Dự đoán tức thời không cần upload file
- Auto-detect mode: Tự động phân tích liên tục
- Điều chỉnh tần suất phân tích (200ms - 5000ms)

**Tính năng**:
- ▶️ Start/Stop camera
- 📸 Capture & Analyze frame
- ✅ Auto-detect (continuous mode)
- Hiển thị confidence và all probabilities
- Select model và pipeline real-time

**API Endpoint**: `/api/webcam_predict` (POST)
- Input: base64 image
- Output: predictions với confidence scores

**Code location**:
- Template: `templates/webcam.html`
- API: `app.py` - Route `/api/webcam_predict`

---

### 3. ✅ **Image Enhancement Pipeline**
**Endpoint**: `/api/enhance_image`

**Chức năng**: Tự động cải thiện chất lượng ảnh trước khi dự đoán

**Các bước xử lý**:
1. **Denoise**: Giảm nhiễu với `cv2.fastNlMeansDenoisingColored`
2. **Brightness/Contrast**: Auto-adjust bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. **Sharpen**: Tăng độ sắc nét với convolution kernel

**Parameters**:
- `denoise`: boolean (default: true)
- `sharpen`: boolean (default: true)
- `adjust_brightness`: boolean (default: true)

**Code location**:
- Backend: `utils.py` - `enhance_image_quality()`
- API: `app.py` - Route `/api/enhance_image`

---

### 4. ✅ **Image Quality Check**
**Endpoint**: `/api/check_quality`

**Chức năng**: Đánh giá chất lượng ảnh trước khi dự đoán

**Các tiêu chí kiểm tra**:
- ✓ **Size**: Kích thước tối thiểu 100x100 pixels
- ✓ **Blur**: Laplacian variance để phát hiện ảnh mờ
- ✓ **Brightness**: Độ sáng trung bình (40-220)
- ✓ **Contrast**: Độ tương phản (std deviation)

**Output**:
```json
{
  "quality": {
    "is_good": true/false,
    "score": 0-100,
    "issues": ["Image too dark", ...],
    "recommendation": "..."
  },
  "image_info": {
    "width": 1024,
    "height": 768,
    "size_kb": 250.5
  }
}
```

**Code location**:
- Backend: `utils.py` - `check_image_quality()`
- API: `app.py` - Route `/api/check_quality`

---

### 5. ✅ **Leaf Region Detection**
**Endpoint**: `/api/detect_leaf`

**Chức năng**: Tự động phát hiện và trích xuất vùng lá cà chua

**Kỹ thuật**:
- Color segmentation trong HSV space
- Green color range: H=[25,85], S=[40,255], V=[40,255]
- Morphological operations để làm sạch mask
- Bounding box detection

**Output**:
- `leaf_detected`: boolean
- `bbox`: {x, y, width, height}
- `image_with_bbox`: Ảnh gốc với bbox màu xanh
- `leaf_image`: Ảnh lá đã crop

**Code location**:
- Backend: `utils.py` - `detect_leaf_region()`
- API: `app.py` - Route `/api/detect_leaf`

---

### 6. ✅ **Batch Image Processing (Enhanced)**
**Route**: `/batch_predict`

**Chức năng**: Upload và xử lý nhiều ảnh cùng lúc (max 10 ảnh)

**Features**:
- Grid layout với cards đẹp mắt
- Hiển thị preview thumbnail
- Status badges (Success/Warning/Rejected)
- Expandable details cho từng ảnh
- Link đến full prediction detail

**Template**: `templates/batch_result.html`

---

## 🛠️ Technical Stack

### Libraries Used:
- **OpenCV** (cv2): Image processing, color space conversion, morphological ops
- **TensorFlow/Keras**: Deep learning models, Grad-CAM
- **NumPy**: Array operations
- **PIL**: Image handling
- **Flask**: Web framework

### Key Algorithms:
1. **Grad-CAM**: Visualization of CNN decisions
2. **CLAHE**: Adaptive histogram equalization
3. **Laplacian variance**: Blur detection
4. **HSV segmentation**: Color-based detection
5. **Morphological operations**: Mask refinement

---

## 📊 API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/gradcam` | POST | Generate disease localization heatmap |
| `/api/enhance_image` | POST | Enhance image quality |
| `/api/check_quality` | POST | Check image quality metrics |
| `/api/detect_leaf` | POST | Detect and extract leaf region |
| `/api/webcam_predict` | POST | Real-time webcam prediction |
| `/webcam` | GET | Webcam detection page |

---

## 🎨 UI/UX Improvements

### Navigation:
- Added 🎥 **Webcam** link to navigation bar

### Result Page:
- Added "🔥 Xem vùng bệnh (Heatmap)" button
- Modal popup for heatmap visualization
- Gradient purple button styling

### Webcam Page:
- Modern two-column layout (camera + result)
- Real-time status indicators (green pulse when active)
- Auto-detect toggle with interval control
- Settings panel for model/pipeline selection

---

## 🚀 Usage Examples

### 1. Generate Heatmap:
```python
# Via API
import requests

with open('tomato_leaf.jpg', 'rb') as f:
    files = {'file': f}
    data = {'model': 'VGG19', 'pipeline': 'average_hsv'}
    response = requests.post('http://localhost:5000/api/gradcam', 
                           files=files, data=data)
    result = response.json()
    heatmap_base64 = result['heatmap_image']
```

### 2. Real-time Webcam:
```javascript
// Capture and predict
const canvas = document.getElementById('canvas');
const imageData = canvas.toDataURL('image/jpeg');

const response = await fetch('/api/webcam_predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    image: imageData,
    model: 'VGG19',
    pipeline: 'average_hsv'
  })
});

const data = await response.json();
console.log(data.prediction);
```

### 3. Quality Check:
```python
# Check before prediction
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/check_quality',
                           files={'file': f})
    quality = response.json()['quality']
    
    if quality['is_good']:
        # Proceed with prediction
        ...
    else:
        print(f"Issues: {quality['issues']}")
```

---

## ⚡ Performance Considerations

1. **Grad-CAM**: Takes ~1-2s per image (depends on model size)
2. **Webcam**: Optimized for real-time with adjustable intervals
3. **Image Enhancement**: ~500ms per image
4. **Quality Check**: ~50ms per image (fast)
5. **Leaf Detection**: ~100-200ms per image

---

## 🔮 Future Enhancements

### Planned:
- [ ] Multi-disease detection (multi-label)
- [ ] Disease progression tracking (time-series)
- [ ] Mobile app with TFLite
- [ ] Drone image processing
- [ ] 3D leaf reconstruction

### Advanced CV:
- [ ] Semantic segmentation (U-Net/Mask R-CNN)
- [ ] Object detection (YOLO for counting leaves)
- [ ] Disease severity scoring
- [ ] Leaf counting and area measurement

---

## 📝 Notes

- All CV functions are in `utils.py` for modularity
- API endpoints follow RESTful conventions
- Error handling with try-catch blocks
- Logging for debugging
- Base64 encoding for image transfer
- Responsive design for mobile devices

---

## 🎓 Educational Value

These CV features demonstrate:
- **Explainable AI**: Grad-CAM shows what model "sees"
- **Real-time inference**: WebRTC + TensorFlow
- **Image preprocessing**: Standard CV pipeline
- **Quality assurance**: Automated QA checks
- **User experience**: Interactive visualizations

Perfect for understanding practical computer vision applications in agriculture! 🌱
