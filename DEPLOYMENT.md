# Hướng dẫn triển khai ứng dụng

## 📋 Yêu cầu hệ thống

### Phần cứng tối thiểu
- **CPU**: Intel Core i5 hoặc tương đương
- **RAM**: 8GB (khuyến nghị 16GB để chạy nhiều models)
- **Ổ cứng**: 10GB khả dụng
- **GPU**: Không bắt buộc (nhưng cải thiện tốc độ prediction)

### Phần mềm
- **Python**: 3.8 - 3.11 (khuyến nghị 3.10)
- **pip**: Phiên bản mới nhất
- **Git**: Để clone repository
- **Browser**: Chrome, Firefox, hoặc Edge (version mới)

---

## 🚀 Triển khai môi trường Development

### Bước 1: Clone repository
```bash
git clone <repository-url>
cd web_tomato
```

### Bước 2: Tạo Python virtual environment
**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Lưu ý cho Windows:** Nếu gặp lỗi với `opencv-python-headless`, thử:
```powershell
pip install opencv-python==4.9.0.80
```

### Bước 4: Cấu hình environment variables
```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Chỉnh sửa file `.env`:
```env
SECRET_KEY=your-random-secret-key-here
ADMIN_USERNAME=admin
ADMIN_PASSWORD=strong-password-123
MAX_LOADED_MODELS=2
LOG_LEVEL=INFO
```

### Bước 5: Chuẩn bị dữ liệu mẫu
Đảm bảo các folder và file sau tồn tại:

```
data/
├── tomato_answer_question.xlsx  # Dataset Q&A (605 câu hỏi)
└── sample_features.pkl          # Sẽ được tạo ở bước 6

static/images/
├── tomato_samples/              # Ảnh lá cà chua (positive samples)
└── not_tomato_samples/          # Ảnh không phải lá cà chua (negative samples)

model/
├── average_hsv/                 # Models trained với pipeline average_hsv
├── median_cmyk/                 # Models trained với pipeline median_cmyk
├── median_hsi/                  # Models trained với pipeline median_hsi
├── noise_cmyk/                  # Models trained với pipeline noise_cmyk
└── noise_hsi/                   # Models trained với pipeline noise_hsi
```

### Bước 6: Build sample features
Chạy script để tính toán đặc trưng của ảnh mẫu:
```bash
cd tomato
python build_sample_features.py
```

Output mong đợi:
```
[INFO] Đang xử lý thư mục positive...
Processing positive samples: 100%|████████████| 352/352
[INFO] Đang xử lý thư mục negative...
Processing negative samples: 100%|████████████| 104/104
[INFO] Đã lưu sample features: 352 positive, 104 negative
```

### Bước 7: Khởi chạy application
```bash
python app.py
```

Application sẽ chạy tại: **http://localhost:5000**

Trình duyệt sẽ tự động mở sau 1.5 giây (có thể tắt bằng `AUTO_OPEN_BROWSER=false` trong `.env`).

---

## 🌐 Triển khai Production (Linux Server)

### Yêu cầu bổ sung
- **Nginx**: Web server reverse proxy
- **Gunicorn**: WSGI HTTP Server
- **Supervisor**: Process manager (tùy chọn)

### Bước 1: Cài đặt system dependencies
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip nginx -y
```

### Bước 2: Clone và cài đặt application
```bash
cd /var/www
sudo git clone <repository-url> tomato_app
cd tomato_app
sudo chown -R $USER:$USER /var/www/tomato_app
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Cấu hình production environment
```bash
cp .env.example .env
nano .env
```

Chỉnh sửa cho production:
```env
SECRET_KEY=<generate-strong-random-key>
FLASK_ENV=production
FLASK_DEBUG=False
LOG_LEVEL=WARNING
AUTO_OPEN_BROWSER=false
SESSION_COOKIE_SECURE=True  # Nếu dùng HTTPS
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<strong-secure-password>
MAX_LOADED_MODELS=2
```

**Tạo SECRET_KEY ngẫu nhiên:**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Bước 4: Chuẩn bị dữ liệu và build features
```bash
cd tomato
python build_sample_features.py
cd ..
```

### Bước 5: Cấu hình Gunicorn
Tạo file `gunicorn_config.py`:
```python
import multiprocessing

bind = "127.0.0.1:8000"
workers = 2  # Giảm nếu RAM ít
worker_class = "sync"
timeout = 120
keepalive = 5
accesslog = "/var/www/tomato_app/logs/gunicorn_access.log"
errorlog = "/var/www/tomato_app/logs/gunicorn_error.log"
loglevel = "info"
```

### Bước 6: Cấu hình Nginx
Tạo file `/etc/nginx/sites-available/tomato_app`:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Thay bằng domain của bạn

    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }

    location /static {
        alias /var/www/tomato_app/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/tomato_app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Bước 7: Cấu hình systemd service
Tạo file `/etc/systemd/system/tomato_app.service`:
```ini
[Unit]
Description=Tomato Disease Detection Web Application
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/tomato_app/tomato
Environment="PATH=/var/www/tomato_app/venv/bin"
ExecStart=/var/www/tomato_app/venv/bin/gunicorn -c /var/www/tomato_app/gunicorn_config.py app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable và start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tomato_app
sudo systemctl start tomato_app
sudo systemctl status tomato_app
```

### Bước 8: Cấu hình HTTPS (khuyến nghị)
Sử dụng Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

Certbot sẽ tự động cấu hình HTTPS redirect.

---

## 📊 Monitoring và Logging

### Xem logs
```bash
# Application logs
tail -f /var/www/tomato_app/tomato/logs/app.log
tail -f /var/www/tomato_app/tomato/logs/error.log

# Gunicorn logs
tail -f /var/www/tomato_app/logs/gunicorn_access.log
tail -f /var/www/tomato_app/logs/gunicorn_error.log

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Systemd service logs
sudo journalctl -u tomato_app -f
```

### Kiểm tra cache statistics
```bash
curl http://localhost:8000/api/cache_stats
```

### Restart service
```bash
sudo systemctl restart tomato_app
```

---

## 🔧 Troubleshooting

### Lỗi: "Out of Memory"
**Giải pháp:**
1. Giảm `MAX_LOADED_MODELS` trong `.env` (ví dụ: từ 2 xuống 1)
2. Giảm số Gunicorn workers trong `gunicorn_config.py`
3. Tăng RAM server hoặc thêm swap space

### Lỗi: "Model not found"
**Kiểm tra:**
```bash
ls -la /var/www/tomato_app/model/average_hsv/
```
Đảm bảo các file `.keras` tồn tại.

### Lỗi: "sample_features.pkl not found"
**Giải pháp:**
```bash
cd /var/www/tomato_app/tomato
source ../venv/bin/activate
python build_sample_features.py
```

### Lỗi: "Permission denied"
**Giải pháp:**
```bash
sudo chown -R www-data:www-data /var/www/tomato_app
sudo chmod -R 755 /var/www/tomato_app
```

### Lỗi: "502 Bad Gateway" (Nginx)
**Kiểm tra:**
1. Gunicorn có đang chạy không:
   ```bash
   sudo systemctl status tomato_app
   ```
2. Kiểm tra port binding:
   ```bash
   sudo netstat -tlnp | grep 8000
   ```
3. Xem logs để tìm lỗi cụ thể

---

## 🔐 Bảo mật Production

### Checklist bảo mật
- ✅ Đổi `ADMIN_PASSWORD` mạnh (ít nhất 12 ký tự, kết hợp chữ/số/ký tự đặc biệt)
- ✅ Sử dụng HTTPS với SSL certificate (Let's Encrypt)
- ✅ Set `SESSION_COOKIE_SECURE=True` trong `.env`
- ✅ Cấu hình firewall:
  ```bash
  sudo ufw allow 80/tcp
  sudo ufw allow 443/tcp
  sudo ufw enable
  ```
- ✅ Giới hạn rate limiting (có thể dùng Nginx limit_req)
- ✅ Backup định kỳ:
  - Database: `data/chat_logs.jsonl`, `data/prediction_history.jsonl`
  - Sample features: `data/sample_features.pkl`
  - Uploaded images: `static/uploaded/`, `static/feedback/`

### Backup script mẫu
```bash
#!/bin/bash
BACKUP_DIR="/backups/tomato_app"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" \
    /var/www/tomato_app/data/ \
    /var/www/tomato_app/static/uploaded/ \
    /var/www/tomato_app/static/feedback/

# Giữ chỉ 7 backup gần nhất
ls -t "$BACKUP_DIR"/data_*.tar.gz | tail -n +8 | xargs -r rm
```

Thêm vào crontab để chạy hàng ngày:
```bash
sudo crontab -e
# Thêm dòng:
0 2 * * * /path/to/backup_script.sh
```

---

## 📈 Performance Tuning

### Tối ưu hóa RAM
- Sử dụng `MAX_LOADED_MODELS=1` nếu RAM < 8GB
- Clear cache định kỳ: `POST /api/clear_cache`

### Tối ưu hóa CPU
- Tăng Gunicorn workers nếu CPU nhiều cores:
  ```python
  workers = min(multiprocessing.cpu_count() * 2 + 1, 4)
  ```

### Tối ưu hóa Disk I/O
- Mount `/var/www/tomato_app/static/uploaded/` lên SSD nếu có
- Cấu hình log rotation:
  ```bash
  sudo nano /etc/logrotate.d/tomato_app
  ```
  ```
  /var/www/tomato_app/tomato/logs/*.log {
      daily
      rotate 7
      compress
      delaycompress
      missingok
      notifempty
  }
  ```

---

## 🔐 Bảo mật Admin Panel

### HTTP Basic Authentication

Tất cả routes admin yêu cầu xác thực:

**Protected Routes:**
- `/admin/feedback` - Quản lý feedback
- `/admin/export_chat` - Xuất log chat
- `/admin/feedback_action` - Xử lý feedback
- `/admin/reload_samples` - Reload sample cache
- `/admin/rebuild_samples` - Rebuild features
- `/api/cache_stats` - Cache statistics
- `/api/clear_cache` - Xóa cache

### Cấu hình Authentication

**1. Setup credentials trong `.env`:**
```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your-strong-password-here
```

⚠️ **QUAN TRỌNG:** Đổi password ngay, không dùng mặc định!

**Khuyến nghị password mạnh:**
```env
ADMIN_PASSWORD=TomatoAI@2025#SecurePass!
```

Tiêu chí:
- Tối thiểu 12 ký tự
- Có chữ hoa, chữ thường, số, ký tự đặc biệt
- Không dùng từ điển

**2. Đăng nhập:**

Khi truy cập admin URLs, browser sẽ hiện popup xác thực:
- Nhập username và password từ `.env`
- Browser lưu session → không cần đăng nhập lại

**3. Đăng xuất:**
- Chrome/Edge: Xóa cookies hoặc đóng tabs
- Firefox: `Ctrl+Shift+Del` → Xóa Active Logins
- Hoặc: Đổi credentials trong `.env` và restart

### Kiểm tra bảo mật

**Test unauthorized access:**
```bash
curl http://localhost:5000/admin/feedback
# Expected: 401 Unauthorized
```

**Test with wrong credentials:**
```bash
curl -u wrong:wrong http://localhost:5000/admin/feedback
# Expected: 401 Unauthorized
```

**Test with correct credentials:**
```bash
curl -u admin:your-password http://localhost:5000/admin/feedback
# Expected: 200 OK
```

### Bảo mật Production

**1. HTTPS (Bắt buộc!)**

HTTP Basic Auth qua HTTP = **mật khẩu cleartext**

Giải pháp:
- Nginx/Apache với SSL certificate
- Cloudflare (free SSL)
- Let's Encrypt:
  ```bash
  sudo certbot --nginx -d your-domain.com
  ```

**2. IP Whitelist (Khuyên dùng)**

Chỉ cho phép admin từ IP cụ thể:

```python
ALLOWED_ADMIN_IPS = ['192.168.1.100', '10.0.0.5']

@requires_admin_auth
def admin_feedback():
    if request.remote_addr not in ALLOWED_ADMIN_IPS:
        abort(403)
    # ...
```

**3. Rate Limiting**

Ngăn brute-force:
```bash
pip install Flask-Limiter
```

```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/admin/feedback')
@limiter.limit("10 per minute")
@requires_admin_auth
def admin_feedback():
    # ...
```

**4. Session Timeout**

```python
from datetime import timedelta
app.permanent_session_lifetime = timedelta(minutes=30)
```

**5. Đổi username mặc định**

```env
ADMIN_USERNAME=tomato_admin_2025  # Không dùng 'admin'
ADMIN_PASSWORD=Very$trong@Password123!
```

### Monitoring & Logs

Mọi unauthorized access được log:
```
[WARNING] Unauthorized admin access attempt from 192.168.1.100
```

Xem logs:
```bash
# Windows
Get-Content logs\app.log -Tail 50 | Select-String "Unauthorized"

# Linux
tail -f logs/app.log | grep "Unauthorized"
```

### Security Checklist Production

- [ ] Đổi password mặc định trong `.env`
- [ ] Đổi username khác 'admin'
- [ ] Enable HTTPS (production)
- [ ] Thêm IP whitelist (tùy chọn)
- [ ] Setup rate limiting
- [ ] Monitor logs thường xuyên
- [ ] Backup `.env` an toàn
- [ ] **KHÔNG commit `.env` vào Git!** (thêm vào `.gitignore`)

### Khôi phục Access

Nếu quên password:
1. Stop Flask app
2. Sửa `.env`: `ADMIN_PASSWORD=temporary123`
3. Restart app
4. Đăng nhập, đổi password mạnh ngay

---

## ✅ Kiểm tra sau deployment

1. **Truy cập homepage**: http://your-domain.com
2. **Upload ảnh test**: Kiểm tra prediction hoạt động
3. **Test chatbot**: Hỏi "Bệnh sớm là gì?"
4. **Test admin panel**: http://your-domain.com/admin/feedback
5. **Kiểm tra logs**: Không có ERROR trong logs
6. **Test performance**: Upload nhiều ảnh liên tục
7. **Kiểm tra HTTPS**: Force HTTPS redirect

---

## 📞 Support

Nếu gặp vấn đề trong quá trình triển khai:
1. Kiểm tra logs ở section "Monitoring và Logging"
2. Tham khảo "Troubleshooting" section
3. Kiểm tra GitHub Issues của project

**Chúc bạn triển khai thành công!** 🎉
