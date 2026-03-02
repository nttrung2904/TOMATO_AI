# Hướng dẫn lấy OpenWeatherMap API Key

## Bước 1: Đăng ký tài khoản

1. Truy cập: https://home.openweathermap.org/users/sign_up
2. Điền thông tin:
   - Email
   - Username  
   - Password
3. Click **Create Account**
4. Check email để verify tài khoản (click link xác nhận)

## Bước 2: Lấy API Key

1. Đăng nhập: https://home.openweathermap.org/
2. Vào **API Keys** tab: https://home.openweathermap.org/api_keys
3. **Default API key** đã được tạo sẵn (hoặc click **Generate** để tạo mới)
4. Copy API key (dạng: `fc1a7373afe4ebeb0c2a13e8f7f1276e`)

## Bước 3: Đợi kích hoạt

⏰ **Quan trọng:** API key mới cần **10-20 phút** để được kích hoạt!

Trong lúc đợi:
- Status hiển thị có thể là: "Processing" hoặc "Active"
- Nếu dùng ngay sẽ bị lỗi 401 Unauthorized

## Bước 4: Test API Key

### Test trên browser:
```
https://api.openweathermap.org/data/2.5/weather?q=Hanoi&appid=YOUR_API_KEY_HERE&units=metric
```

Thay `YOUR_API_KEY_HERE` bằng API key của bạn.

**Kết quả mong đợi:** JSON với dữ liệu thời tiết Hà Nội

**Nếu lỗi 401:** API key chưa active, đợi thêm vài phút

### Test bằng PowerShell:
```powershell
curl "https://api.openweathermap.org/data/2.5/weather?q=Hanoi&appid=YOUR_API_KEY&units=metric"
```

## Bước 5: Cập nhật vào ứng dụng

1. Mở file `.env` trong thư mục gốc
2. Tìm dòng: `OPENWEATHER_API_KEY=...`
3. Thay bằng API key mới:
   ```
   OPENWEATHER_API_KEY=your_new_valid_api_key_here
   ```
4. Lưu file
5. **Restart Flask server** (Stop và Run lại)

## Troubleshooting

### ❌ Lỗi 401: Invalid API key
**Nguyên nhân:** API key chưa được kích hoạt hoặc không đúng
**Giải pháp:** 
- Đợi 10-20 phút sau khi tạo
- Verify key tại https://home.openweathermap.org/api_keys
- Tạo key mới nếu cần

### ❌ Lỗi 429: Too many requests
**Nguyên nhân:** Vượt quá giới hạn free tier (1000 calls/day)
**Giải pháp:**
- Chờ đến ngày hôm sau (reset quota)
- Hoặc upgrade plan (trả phí)

### ❌ Widget không hiển thị sau khi thêm key
**Giải pháp:**
1. Kiểm tra console log (F12 > Console)
2. Verify Flask đã restart
3. Clear browser cache (Ctrl+F5)
4. Check file .env có đúng format (không có dấu cách, quotes thừa)

## Free Tier Limits

✅ **Miễn phí vĩnh viễn:**
- 1,000 API calls/day
- 60 calls/minute
- Current weather + 5 day forecast
- 16 day forecast (coming soon)

⚠️ **Lưu ý:**
- Widget auto-refresh 10 phút = 144 calls/day/user
- Cache 10 phút để tiết kiệm quota
- Đủ cho ~7 concurrent users

## Thông tin thêm

- Tài liệu API: https://openweathermap.org/api
- FAQ: https://openweathermap.org/faq
- Pricing: https://openweathermap.org/price
