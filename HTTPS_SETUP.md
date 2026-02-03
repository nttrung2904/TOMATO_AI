# Hướng dẫn Cấu hình HTTPS cho Ứng dụng Tomato Disease Detection

## 📋 Tổng quan

Có 3 phương pháp chính để thêm HTTPS vào ứng dụng Flask:

1. **Self-signed Certificate** (Development) - Nhanh nhất, cho môi trường phát triển
2. **Let's Encrypt Certificate** (Production) - Miễn phí, tự động gia hạn
3. **Reverse Proxy với Nginx/Apache** (Production) - Khuyến nghị nhất cho production

---

## 🔧 Phương pháp 1: Self-signed Certificate (Development)

### Bước 1: Tạo Self-signed Certificate

```powershell
# Cài đặt OpenSSL nếu chưa có
# Download từ: https://slproweb.com/products/Win32OpenSSL.html

# Tạo thư mục certs
New-Item -ItemType Directory -Force -Path "certs"

# Tạo self-signed certificate (valid 365 ngày)
openssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365 -subj "/CN=localhost"
```

**Hoặc dùng PowerShell (Windows):**

```powershell
# Tạo self-signed certificate bằng PowerShell
$cert = New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "Cert:\CurrentUser\My" -NotAfter (Get-Date).AddYears(1)

# Export certificate
$pwd = ConvertTo-SecureString -String "password123" -Force -AsPlainText
New-Item -ItemType Directory -Force -Path "certs"
Export-PfxCertificate -Cert $cert -FilePath "certs\cert.pfx" -Password $pwd

# Convert PFX to PEM (cần OpenSSL)
openssl pkcs12 -in certs\cert.pfx -out certs\cert.pem -nodes -passin pass:password123
openssl pkcs12 -in certs\cert.pfx -out certs\key.pem -nocerts -nodes -passin pass:password123
```

### Bước 2: Cập nhật `.gitignore`

```
# SSL Certificates
certs/
*.pem
*.pfx
*.key
```

### Bước 3: Thêm biến môi trường vào `.env`

```ini
# HTTPS Configuration
USE_HTTPS=true
SSL_CERT_PATH=certs/cert.pem
SSL_KEY_PATH=certs/key.pem
```

### Bước 4: Code đã được cập nhật trong `app.py`

Code tự động detect và sử dụng HTTPS nếu có certificate files.

### Bước 5: Chạy ứng dụng

```powershell
cd tomato
python app.py
```

Truy cập: `https://localhost:5000`

**Lưu ý:** Trình duyệt sẽ cảnh báo về certificate không tin cậy. Chọn "Advanced" → "Proceed to localhost" để tiếp tục.

---

## 🌐 Phương pháp 2: Let's Encrypt (Production với Domain)

### Yêu cầu:
- Domain name đã trỏ về server của bạn
- Server Linux (Ubuntu/Debian)
- Port 80 và 443 mở

### Bước 1: Cài đặt Certbot

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install certbot python3-certbot-nginx -y

# CentOS/RHEL
sudo yum install certbot python3-certbot-nginx -y
```

### Bước 2: Lấy Certificate

```bash
# Đảm bảo domain đã trỏ về IP server
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Certificate sẽ được lưu tại:
# /etc/letsencrypt/live/yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

### Bước 3: Cập nhật `.env` trên production server

```ini
USE_HTTPS=true
SSL_CERT_PATH=/etc/letsencrypt/live/yourdomain.com/fullchain.pem
SSL_KEY_PATH=/etc/letsencrypt/live/yourdomain.com/privkey.pem
```

### Bước 4: Thiết lập Auto-renewal

```bash
# Test renewal
sudo certbot renew --dry-run

# Cron job tự động gia hạn (đã có sẵn sau khi cài certbot)
sudo systemctl status certbot.timer
```

---

## 🚀 Phương pháp 3: Nginx Reverse Proxy (Khuyến nghị cho Production)

### Ưu điểm:
- Hiệu suất tốt nhất
- Quản lý SSL tập trung
- Load balancing, caching
- Tách biệt web server và app server

### Bước 1: Cài đặt Nginx

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nginx -y

# Start Nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

### Bước 2: Lấy Let's Encrypt Certificate

```bash
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

### Bước 3: Cấu hình Nginx

Tạo file `/etc/nginx/sites-available/tomato-app`:

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name yourdomain.com www.yourdomain.com;
    
    # Redirect all HTTP traffic to HTTPS
    return 301 https://$server_name$request_uri;
}

# HTTPS Server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL Certificate Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_trusted_certificate /etc/letsencrypt/live/yourdomain.com/chain.pem;

    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Max upload size (phải khớp với Flask MAX_CONTENT_LENGTH)
    client_max_body_size 16M;

    # Logging
    access_log /var/log/nginx/tomato-app-access.log;
    error_log /var/log/nginx/tomato-app-error.log;

    # Static files
    location /static {
        alias /path/to/web_tomato/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Proxy to Flask app
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        proxy_buffering off;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Bước 4: Enable site và restart Nginx

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/tomato-app /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### Bước 5: Chạy Flask app (không cần HTTPS trong app)

```bash
# Flask chỉ cần chạy HTTP vì Nginx sẽ handle HTTPS
cd /path/to/web_tomato/tomato
python app.py
```

### Bước 6: Thiết lập Systemd Service (tùy chọn)

Tạo file `/etc/systemd/system/tomato-app.service`:

```ini
[Unit]
Description=Tomato Disease Detection Flask App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/web_tomato/tomato
Environment="PATH=/path/to/web_tomato/venv/bin"
ExecStart=/path/to/web_tomato/venv/bin/python app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable tomato-app
sudo systemctl start tomato-app
sudo systemctl status tomato-app
```

---

## 🔒 Bổ sung: Flask-Talisman (Force HTTPS)

### Cài đặt

```powershell
pip install flask-talisman
```

### Thêm vào requirements.txt

```
flask-talisman>=1.1.0
```

### Sử dụng trong code (đã thêm vào app.py)

Flask-Talisman tự động:
- Force HTTPS
- Thêm security headers
- Content Security Policy (CSP)
- Clickjacking protection

---

## 📊 So sánh các phương pháp

| Tiêu chí | Self-signed | Let's Encrypt | Nginx + Let's Encrypt |
|----------|-------------|---------------|----------------------|
| **Độ khó** | ⭐ Dễ | ⭐⭐ Trung bình | ⭐⭐⭐ Khó |
| **Chi phí** | Miễn phí | Miễn phí | Miễn phí |
| **Bảo mật** | Thấp (dev only) | Cao | Rất cao |
| **Hiệu năng** | Trung bình | Trung bình | Cao |
| **Production** | ❌ Không | ✅ Được | ✅ Khuyến nghị |
| **Auto-renewal** | ❌ Không | ✅ Có | ✅ Có |
| **Cảnh báo browser** | ⚠️ Có | ✅ Không | ✅ Không |

---

## 🛠️ Khắc phục sự cố

### Lỗi: "Certificate verify failed"

```python
# Chỉ dùng cho development/testing
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
```

### Lỗi: Port 443 đã được sử dụng

```bash
# Kiểm tra process đang dùng port 443
sudo netstat -tulpn | grep :443
# hoặc
sudo lsof -i :443

# Kill process nếu cần
sudo kill -9 <PID>
```

### Lỗi: Permission denied khi bind port 443

```bash
# Linux: Cho phép Python bind port < 1024
sudo setcap CAP_NET_BIND_SERVICE=+eip /path/to/python

# Hoặc chạy với sudo (không khuyến nghị)
sudo python app.py
```

### Nginx không start sau cấu hình SSL

```bash
# Kiểm tra lỗi
sudo nginx -t

# Xem log chi tiết
sudo tail -f /var/log/nginx/error.log

# Kiểm tra certificate files có tồn tại không
sudo ls -l /etc/letsencrypt/live/yourdomain.com/
```

---

## ✅ Checklist Triển khai Production

- [ ] Domain đã trỏ về IP server
- [ ] Firewall mở port 80, 443
- [ ] Cài đặt Nginx
- [ ] Lấy Let's Encrypt certificate
- [ ] Cấu hình Nginx với SSL
- [ ] Thiết lập auto-renewal cho certificate
- [ ] Cấu hình systemd service cho Flask app
- [ ] Test HTTPS hoạt động: `curl -I https://yourdomain.com`
- [ ] Test auto-redirect HTTP → HTTPS
- [ ] Kiểm tra SSL rating: https://www.ssllabs.com/ssltest/
- [ ] Thiết lập monitoring và logging
- [ ] Backup configuration files

---

## 📚 Tài liệu tham khảo

- [Flask SSL Context](https://flask.palletsprojects.com/en/2.3.x/deploying/wsgi-standalone/#ssl-context)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
- [Nginx SSL Configuration](https://nginx.org/en/docs/http/configuring_https_servers.html)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [SSL Labs Server Test](https://www.ssllabs.com/ssltest/)
