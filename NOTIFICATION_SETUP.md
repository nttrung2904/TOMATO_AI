# Hướng Dẫn Thiết Lập Hệ Thống Thông Báo

## 1. Tổng Quan

Dự án hỗ trợ 2 loại thông báo:
- **Email Notifications**: Gửi email tự động cho đơn hàng, thanh toán, đăng ký
- **In-App Notifications**: Thông báo trên giao diện web với biểu tượng chuông

---

## 2. Cài Đặt Email Notifications

### 2.1. Cấu Hình SMTP (Gmail)

#### Bước 1: Tạo App Password cho Gmail
1. Đăng nhập vào Google Account: https://myaccount.google.com
2. Chọn **Security** → **2-Step Verification** (bật nếu chưa có)
3. Cuộn xuống **App passwords**
4. Chọn app: **Mail**, device: **Other (Custom name)**
5. Nhập tên: `Tomato Web App`
6. Copy mật khẩu 16 ký tự (format: `xxxx xxxx xxxx xxxx`)

#### Bước 2: Cập Nhật File `.env`
```env
# SMTP Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx  # App password vừa tạo
SMTP_FROM_EMAIL=your-email@gmail.com
SMTP_FROM_NAME=Tomato Care
```

#### Bước 3: Cài Đặt Thư Viện (nếu chưa có)
```bash
pip install flask requests
```

### 2.2. Các Loại Email Tự Động

| Email Type | Trigger | Template |
|------------|---------|----------|
| **Order Confirmation** | Khi khách hàng đặt hàng | `EmailTemplates.order_confirmation()` |
| **Payment Success** | Khi thanh toán thành công | `EmailTemplates.payment_success()` |
| **Payment Failed** | Khi thanh toán thất bại | `EmailTemplates.payment_failed()` |
| **Welcome Email** | Khi đăng ký tài khoản mới | `EmailTemplates.welcome_email()` |

### 2.3. Test Gửi Email

```python
from tomato.notifications import email_service

# Test gửi email đơn giản
success = email_service.send_email(
    to_email="test@example.com",
    subject="Test Email",
    html_content="<h1>Hello from Tomato App!</h1>"
)

if success:
    print("✅ Email sent successfully")
else:
    print("❌ Failed to send email")
```

### 2.4. Tùy Chỉnh Email Templates

Chỉnh sửa file `tomato/notifications.py` để thay đổi template:

```python
@staticmethod
def order_confirmation(customer_name, order_id, order_total, items):
    # Thay đổi màu sắc brand
    primary_color = "#4CAF50"  # Màu xanh lá cây
    
    # Thay đổi logo
    logo_html = '<div style="text-align:center;margin-bottom:20px;">' \
                '<img src="https://your-domain.com/logo.png" alt="Logo" width="150">' \
                '</div>'
    
    # ... custom content
```

---

## 3. Cài Đặt In-App Notifications

### 3.1. Kiến Trúc Hệ Thống

```
User Action → Create Notification → Save to JSONL → Display in UI
                                   → Update Badge Count
```

**Storage**: `data/notifications.jsonl`

**Structure**:
```json
{
  "id": "notif_1234567890",
  "user_id": "user_123",
  "title": "Đơn hàng thành công",
  "message": "Đơn hàng #ORD001 đã được xác nhận",
  "type": "success",
  "link": "/history",
  "read": false,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 3.2. Tạo Thông Báo Trong Code

```python
from tomato.app import create_notification

# Tạo thông báo cho user
create_notification(
    user_id="user_123",
    title="Thanh toán thành công",
    message="Bạn đã thanh toán thành công 150,000 VNĐ",
    notification_type="success",
    link="/history"
)
```

### 3.3. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/notifications` | GET | Lấy danh sách thông báo |
| `/api/notifications/<id>/read` | POST | Đánh dấu 1 thông báo đã đọc |
| `/api/notifications/read_all` | POST | Đánh dấu tất cả đã đọc |

### 3.4. Test API

```bash
# 1. Lấy danh sách thông báo
curl http://localhost:5000/api/notifications

# 2. Đánh dấu đã đọc
curl -X POST http://localhost:5000/api/notifications/notif_1234567890/read

# 3. Đánh dấu tất cả
curl -X POST http://localhost:5000/api/notifications/read_all
```

### 3.5. Frontend Integration

Biểu tượng chuông đã được tích hợp vào `templates/base.html`:

**Features**:
- ✅ Badge hiển thị số thông báo chưa đọc
- ✅ Dropdown với danh sách thông báo
- ✅ Tự động refresh mỗi 2 phút
- ✅ Click vào thông báo → đánh dấu đã đọc
- ✅ Nút "Đánh dấu tất cả là đã đọc"
- ✅ Dark mode support

**Customization**:
```css
/* Thay đổi màu badge trong templates/base.html */
.notification-badge {
  background-color: #ff4444; /* Đỏ mặc định */
}

/* Thay đổi màu icon */
.notification-bell svg {
  fill: #333; /* Màu icon */
}
```

---

## 4. Tích Hợp Thông Báo Vào Flow

### 4.1. Khi Đặt Hàng

File: `tomato/app.py` → Route: `/submit_order`

```python
# Gửi email xác nhận
email_service.send_order_confirmation(
    user_email=user.get('email'),
    customer_name=user.get('fullname'),
    order_id=order_id,
    order_total=total_price,
    items=cart_items
)

# Tạo thông báo trong app
create_notification(
    user_id=user_id,
    title="Đơn hàng mới",
    message=f"Đơn hàng #{order_id} trị giá {total_price:,}đ đã được tạo",
    notification_type="info",
    link="/history"
)
```

### 4.2. Khi Thanh Toán Thành Công

File: `tomato/app.py` → Route: `/payment/vnpay/callback`

```python
# Gửi email thành công
email_service.send_payment_success(
    user_email=order.get('email'),
    customer_name=order.get('customer_name'),
    order_id=order_id,
    amount=order.get('total_price')
)

# Thông báo trong app
create_notification(
    user_id=order.get('user_id'),
    title="Thanh toán thành công",
    message=f"Đơn hàng #{order_id} đã được thanh toán",
    notification_type="success",
    link="/history"
)
```

### 4.3. Khi Đăng Ký Tài Khoản

File: `tomato/app.py` → Route: `/register`

```python
# Gửi email chào mừng
email_service.send_welcome_email(
    user_email=email,
    customer_name=fullname
)

# Thông báo chào mừng
create_notification(
    user_id=new_user_id,
    title="Chào mừng!",
    message="Chào mừng bạn đến với Tomato Care! 🍅",
    notification_type="success",
    link="/profile"
)
```

---

## 5. Testing Checklist

### Email Notifications
- [ ] Cấu hình SMTP trong `.env`
- [ ] Test gửi email đơn giản
- [ ] Test email đặt hàng (submit order)
- [ ] Test email thanh toán thành công (VNPay/MoMo)
- [ ] Test email thanh toán thất bại
- [ ] Test email đăng ký tài khoản
- [ ] Kiểm tra email hiển thị đúng trên mobile
- [ ] Kiểm tra email không bị đánh dấu spam

### In-App Notifications
- [ ] Đăng ký tài khoản → kiểm tra badge có xuất hiện
- [ ] Đặt hàng → kiểm tra thông báo mới
- [ ] Thanh toán → kiểm tra thông báo
- [ ] Click vào thông báo → đánh dấu đã đọc → badge giảm
- [ ] Test nút "Đánh dấu tất cả" → badge = 0
- [ ] Test tự động refresh sau 2 phút
- [ ] Test dark mode → UI hiển thị đúng

---

## 6. Troubleshooting

### 6.1. Email Không Gửi Được

**Vấn đề 1: SMTP Authentication Failed**
```
SMTPAuthenticationError: Username and Password not accepted
```

**Giải pháp**:
- Kiểm tra đã bật 2-Step Verification cho Gmail
- Tạo lại App Password
- Đảm bảo không có khoảng trắng trong password
- Kiểm tra `SMTP_USERNAME` đúng email

**Vấn đề 2: Connection Timeout**
```
TimeoutError: [Errno 110] Connection timed out
```

**Giải pháp**:
- Kiểm tra firewall/antivirus có chặn port 587
- Thử đổi `SMTP_PORT=465` và thêm `use_ssl=True`
- Kiểm tra kết nối internet

**Vấn đề 3: Email Vào Spam**

**Giải pháp**:
- Thêm SPF record cho domain
- Sử dụng domain email chính thức (không dùng Gmail cá nhân cho production)
- Tránh từ ngữ spam trong subject ("FREE", "WIN", "CLICK HERE")
- Thêm unsubscribe link

### 6.2. In-App Notifications Không Hiển Thị

**Vấn đề 1: Badge Không Xuất Hiện**

**Kiểm tra**:
- Console browser có lỗi JavaScript?
- User đã đăng nhập chưa? (`session.get('user_id')`)
- API `/api/notifications` trả về data?

**Giải pháp**:
```bash
# 1. Check browser console (F12)
# 2. Test API manually
curl http://localhost:5000/api/notifications

# 3. Check notifications.jsonl
cat data/notifications.jsonl | grep "user_id_của_bạn"
```

**Vấn đề 2: Thông Báo Không Real-time**

**Nguyên nhân**: Hệ thống dùng polling (2 phút/lần), không dùng WebSocket

**Giải pháp nâng cao**: Tích hợp Socket.IO
```bash
pip install flask-socketio
```

### 6.3. Performance Issues

**Vấn đề**: Nhiều thông báo làm file JSONL lớn

**Giải pháp**:
```python
# Tự động xóa thông báo cũ hơn 30 ngày
from datetime import datetime, timedelta

def cleanup_old_notifications():
    cutoff = datetime.now() - timedelta(days=30)
    notifications = load_notifications()
    
    updated = [n for n in notifications 
               if datetime.fromisoformat(n['timestamp']) > cutoff]
    
    save_notifications(updated)
```

---

## 7. Best Practices

### 7.1. Email
- ✅ Sử dụng threading để gửi email không chặn request
- ✅ Có fallback khi SMTP fail (log lỗi, retry queue)
- ✅ Cung cấp plaintext alternative cho HTML email
- ✅ Test email trên nhiều client (Gmail, Outlook, Yahoo)

### 7.2. In-App Notifications
- ✅ Giới hạn số thông báo hiển thị (100 gần nhất)
- ✅ Auto-cleanup thông báo cũ
- ✅ Thêm pagination cho danh sách dài
- ✅ Tối ưu polling interval (2 phút hợp lý)

### 7.3. Security
- ✅ Không expose SMTP password trong logs
- ✅ Validate user_id trước khi tạo notification
- ✅ Sanitize HTML trong notification message
- ✅ Rate limit API endpoints

---

## 8. Production Deployment

### 8.1. Environment Variables
```bash
# Production .env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=noreply@yourdomain.com
SMTP_PASSWORD=<production-app-password>
SMTP_FROM_EMAIL=noreply@yourdomain.com
SMTP_FROM_NAME=Tomato Care Team
```

### 8.2. Monitoring

**Log Email Status**:
```python
import logging

logger = logging.getLogger(__name__)

if email_sent:
    logger.info(f"Email sent to {to_email}: {subject}")
else:
    logger.error(f"Failed to send email to {to_email}")
```

**Track Notification Metrics**:
```python
# Thêm endpoint để monitor
@app.route('/admin/notifications/stats')
def notification_stats():
    notifications = load_all_notifications()
    
    return {
        'total': len(notifications),
        'unread': sum(1 for n in notifications if not n['read']),
        'by_type': {
            'success': sum(1 for n in notifications if n['type'] == 'success'),
            'error': sum(1 for n in notifications if n['type'] == 'error'),
            # ...
        }
    }
```

### 8.3. Backup

```bash
# Backup notifications mỗi ngày
cp data/notifications.jsonl backups/notifications_$(date +%Y%m%d).jsonl
```

---

## 9. Future Enhancements

### 9.1. Push Notifications (Web Push API)
```javascript
// Đăng ký service worker
navigator.serviceWorker.register('/sw.js');

// Request notification permission
Notification.requestPermission();
```

### 9.2. SMS Notifications
```python
# Sử dụng Twilio hoặc SMSAPI.vn
from twilio.rest import Client

def send_sms(phone, message):
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message,
        from_='+84xxxxxxxx',
        to=phone
    )
```

### 9.3. Email Queue (Celery)
```python
from celery import Celery

@celery.task
def send_email_async(to_email, subject, html_content):
    email_service.send_email(to_email, subject, html_content)
```

---

## 10. Support & Contact

- **Documentation**: This file
- **Email Integration**: `tomato/notifications.py`
- **In-App Notifications**: `tomato/app.py` (search for `create_notification`)
- **Frontend**: `templates/base.html` (notification bell)

**Common Commands**:
```bash
# Test email configuration
python -c "from tomato.notifications import email_service; print(email_service.test_connection())"

# Count notifications
wc -l data/notifications.jsonl

# View recent notifications
tail -n 10 data/notifications.jsonl

# Clear all notifications (CAREFUL!)
echo "" > data/notifications.jsonl
```

---

**Chúc bạn triển khai thành công! 🍅✨**
