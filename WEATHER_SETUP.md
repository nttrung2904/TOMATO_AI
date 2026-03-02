# Weather Integration Setup Guide

## Tích hợp Dữ liệu Thời tiết

Tính năng này tích hợp dữ liệu thời tiết từ OpenWeatherMap API để:
- Hiển thị thời tiết hiện tại (nhiệt độ, độ ẩm, gió, v.v.)
- Đánh giá rủi ro bệnh cà chua dựa trên điều kiện thời tiết
- Cung cấp khuyến nghị chăm sóc dựa trên dự báo

## Cài đặt

### 1. Lấy API Key miễn phí

1. Truy cập [OpenWeatherMap](https://openweathermap.org/api)
2. Đăng ký tài khoản miễn phí
3. Sau khi đăng nhập, vào [API Keys](https://home.openweathermap.org/api_keys)
4. Copy API key của bạn

**Free Tier:**
- 1,000 API calls/day
- 60 calls/minute
- Dữ liệu thời tiết hiện tại và dự báo 5 ngày
- Hoàn toàn miễn phí

### 2. Cấu hình API Key

1. Tạo file `.env` trong thư mục gốc (nếu chưa có):
   ```bash
   cp .env.example .env
   ```

2. Thêm API key vào file `.env`:
   ```
   OPENWEATHER_API_KEY=your-api-key-here
   ```

3. Lưu file và restart server Flask

### 3. Verify Installation

Sau khi restart server, widget thời tiết sẽ xuất hiện ở sidebar trang chủ.

Nếu thấy lỗi:
- Check console log để xem thông báo lỗi
- Verify API key đã được set trong `.env`
- Kiểm tra API key còn hạn sử dụng

## Tính năng

### 1. Thời tiết hiện tại
- Nhiệt độ và cảm giác như
- Độ ẩm và áp suất khí quyển
- Tốc độ gió
- Mô tả thời tiết (mây, mưa, nắng, v.v.)
- Thời gian bình minh/hoàng hôn

### 2. Đánh giá rủi ro bệnh
Hệ thống tự động đánh giá rủi ro bệnh dựa trên:

**Mức độ rủi ro thấp (Low):**
- Điều kiện thời tiết bình thường, không thuận lợi cho bệnh

**Mức độ rủi ro trung bình (Medium):**
- Nhiệt độ 20-30°C + Độ ẩm 70-80%
- Cần theo dõi và có biện pháp phòng ngừa

**Mức độ rủi ro cao (High):**
- Nhiệt độ 20-30°C + Độ ẩm >80% (thuận lợi cho bệnh nấm)
- Mưa nhiều + Gió mạnh (phát tán mầm bệnh)
- Nhiệt độ >32°C + Độ ẩm cao (vi khuẩn phát triển)

### 3. Khuyến nghị chăm sóc
Dựa trên điều kiện thời tiết, hệ thống đưa ra khuyến nghị cụ thể:
- Tăng cường thông gió
- Điều chỉnh lịch tưới nước
- Biện pháp che chắn
- Phun thuốc phòng bệnh
- v.v.

### 4. Thay đổi địa điểm
- Mặc định: Hanoi
- Click nút "📍 Đổi địa điểm" để nhập thành phố khác
- Hỗ trợ tên thành phố tiếng Việt và tiếng Anh

## API Endpoints

### GET /api/weather/current
Lấy thời tiết hiện tại

**Query Parameters:**
- `city` (string, optional): Tên thành phố (default: "Hanoi")
- `lat` (float, optional): Vĩ độ
- `lon` (float, optional): Kinh độ

**Response:**
```json
{
  "success": true,
  "weather": {
    "temperature": 25.5,
    "feels_like": 27.0,
    "humidity": 75,
    "wind_speed": 12.5,
    "description": "mây rải rác",
    "icon": "02d",
    "city_name": "Hanoi",
    "disease_risk": {
      "risk_level": "medium",
      "risk_text": "Trung bình",
      "risk_color": "orange",
      "recommendations": [...]
    }
  }
}
```

### GET /api/weather/forecast
Lấy dự báo thời tiết 5 ngày

**Query Parameters:**
- `city` (string, optional): Tên thành phố
- `lat` (float, optional): Vĩ độ
- `lon` (float, optional): Kinh độ
- `days` (int, optional): Số ngày dự báo (default: 5, max: 5)

**Response:**
```json
{
  "success": true,
  "forecast": {
    "city": "Hanoi",
    "country": "VN",
    "forecasts": [
      {
        "date": "2026-02-26",
        "day_name": "Wednesday",
        "temp_min": 18.5,
        "temp_max": 28.0,
        "temp_avg": 23.2,
        "humidity_avg": 72,
        "description": "mây rải rác",
        "rain_total": 0
      },
      ...
    ]
  }
}
```

## Customization

### Thay đổi thành phố mặc định

Sửa trong `weather_widget.html`:
```javascript
let currentCity = localStorage.getItem('weatherCity') || 'Hanoi';
```

### Thay đổi tần suất cập nhật

Mặc định: 10 phút. Sửa trong `weather_widget.html`:
```javascript
// Auto-refresh every 10 minutes
setInterval(() => loadWeather(), 10 * 60 * 1000);
```

### Thêm widget vào trang khác

Add vào template:
```django
{% include 'weather_widget.html' %}
```

## Troubleshooting

### Lỗi: "API key not configured"
- Kiểm tra file `.env` đã có `OPENWEATHER_API_KEY`
- Restart Flask server sau khi thêm key

### Lỗi: "Weather service timeout"
- Check kết nối internet
- OpenWeatherMap API có thể bị chặn bởi firewall/proxy

### Lỗi: "Invalid API key"
- Verify API key từ OpenWeatherMap dashboard
- API key mới cần đợi vài phút để active

### Widget không hiển thị
- Check browser console (F12) để xem lỗi JavaScript
- Verify route `/api/weather/current` hoạt động
- Clear browser cache (Ctrl+F5)

## Cache & Performance

- Weather data được cache 10 phút để giảm API calls
- Widget auto-refresh mỗi 10 phút
- Free tier: 1000 calls/day = ~60 users x 16 refreshes/day

## Ngôn ngữ

Hỗ trợ đa ngôn ngữ (Tiếng Việt/English) thông qua hệ thống translation có sẵn.

Translation keys trong `data/translations.json`:
```json
"weather": {
  "title": "Thời tiết",
  "current_weather": "Thời tiết hiện tại",
  "disease_risk": "Rủi ro bệnh",
  ...
}
```

## Technical Stack

- **Backend**: Flask (Python)
- **API**: OpenWeatherMap API v2.5
- **Frontend**: Vanilla JavaScript + CSS
- **Cache**: In-memory dict (10 min TTL)
- **HTTP Client**: requests library

## License & Attribution

Weather data provided by [OpenWeatherMap](https://openweathermap.org/).

Free tier restrictions apply. See [OpenWeatherMap Pricing](https://openweathermap.org/price) for details.
