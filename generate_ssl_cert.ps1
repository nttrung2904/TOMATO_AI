# Tạo self-signed SSL certificate cho HTTPS development
# Chạy script này trong PowerShell với quyền Administrator

Write-Host "🔒 Tạo Self-Signed SSL Certificate cho Development" -ForegroundColor Green
Write-Host "=" -ForegroundColor Gray

# Tạo thư mục certs nếu chưa có
$certsDir = "certs"
if (-not (Test-Path $certsDir)) {
    New-Item -ItemType Directory -Path $certsDir | Out-Null
    Write-Host "✓ Đã tạo thư mục: $certsDir" -ForegroundColor Green
} else {
    Write-Host "✓ Thư mục đã tồn tại: $certsDir" -ForegroundColor Yellow
}

# Kiểm tra xem OpenSSL có được cài đặt không
$openssl = Get-Command openssl -ErrorAction SilentlyContinue

if ($openssl) {
    Write-Host "`n📝 Sử dụng OpenSSL để tạo certificate..." -ForegroundColor Cyan
    
    # Tạo certificate với OpenSSL
    $certPath = Join-Path $certsDir "cert.pem"
    $keyPath = Join-Path $certsDir "key.pem"
    
    & openssl req -x509 -newkey rsa:4096 -nodes `
        -out $certPath `
        -keyout $keyPath `
        -days 365 `
        -subj "/C=VN/ST=HoChiMinh/L=HoChiMinh/O=TomatoApp/OU=Development/CN=localhost"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Đã tạo certificate thành công!" -ForegroundColor Green
        Write-Host "  - Certificate: $certPath" -ForegroundColor Gray
        Write-Host "  - Private Key: $keyPath" -ForegroundColor Gray
    } else {
        Write-Host "✗ Lỗi khi tạo certificate với OpenSSL" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n⚠️  OpenSSL không được cài đặt" -ForegroundColor Yellow
    Write-Host "📝 Sử dụng PowerShell để tạo certificate..." -ForegroundColor Cyan
    
    # Tạo certificate với PowerShell
    try {
        $cert = New-SelfSignedCertificate `
            -DnsName "localhost", "127.0.0.1" `
            -CertStoreLocation "Cert:\CurrentUser\My" `
            -NotAfter (Get-Date).AddYears(1) `
            -FriendlyName "Tomato App Development Certificate" `
            -KeyUsage DigitalSignature, KeyEncipherment `
            -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.1")
        
        Write-Host "✓ Đã tạo certificate trong Windows Certificate Store" -ForegroundColor Green
        
        # Export certificate
        $pwd = ConvertTo-SecureString -String "temp123" -Force -AsPlainText
        $pfxPath = Join-Path $certsDir "cert.pfx"
        Export-PfxCertificate -Cert $cert -FilePath $pfxPath -Password $pwd | Out-Null
        
        Write-Host "✓ Đã export certificate ra file PFX" -ForegroundColor Green
        
        # Kiểm tra xem có OpenSSL không để convert sang PEM
        if ($openssl) {
            $certPath = Join-Path $certsDir "cert.pem"
            $keyPath = Join-Path $certsDir "key.pem"
            
            & openssl pkcs12 -in $pfxPath -out $certPath -nokeys -nodes -passin pass:temp123
            & openssl pkcs12 -in $pfxPath -out $keyPath -nocerts -nodes -passin pass:temp123
            
            Write-Host "✓ Đã convert sang định dạng PEM" -ForegroundColor Green
            Write-Host "  - Certificate: $certPath" -ForegroundColor Gray
            Write-Host "  - Private Key: $keyPath" -ForegroundColor Gray
            
            # Xóa file PFX
            Remove-Item $pfxPath -Force
        } else {
            Write-Host "`n⚠️  Cần cài OpenSSL để convert sang PEM format" -ForegroundColor Yellow
            Write-Host "Download từ: https://slproweb.com/products/Win32OpenSSL.html" -ForegroundColor Cyan
            Write-Host "`nSau khi cài OpenSSL, chạy lệnh sau:" -ForegroundColor Yellow
            Write-Host "  openssl pkcs12 -in $pfxPath -out certs/cert.pem -nokeys -nodes -passin pass:temp123" -ForegroundColor Gray
            Write-Host "  openssl pkcs12 -in $pfxPath -out certs/key.pem -nocerts -nodes -passin pass:temp123" -ForegroundColor Gray
        }
    } catch {
        Write-Host "✗ Lỗi: $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n" -NoNewline
Write-Host "=" -ForegroundColor Gray
Write-Host "✅ Hoàn tất!" -ForegroundColor Green
Write-Host "`nĐể bật HTTPS, thêm vào file .env:" -ForegroundColor Cyan
Write-Host "  USE_HTTPS=true" -ForegroundColor White
Write-Host "  SSL_CERT_PATH=certs/cert.pem" -ForegroundColor White
Write-Host "  SSL_KEY_PATH=certs/key.pem" -ForegroundColor White
Write-Host "`nSau đó chạy:" -ForegroundColor Cyan
Write-Host "  python tomato/app.py" -ForegroundColor White
Write-Host "`nTruy cập: https://localhost:5000" -ForegroundColor Green
Write-Host "`n⚠️  Lưu ý: Trình duyệt sẽ cảnh báo về certificate không tin cậy." -ForegroundColor Yellow
Write-Host "   Chọn 'Advanced' → 'Proceed to localhost' để tiếp tục." -ForegroundColor Yellow
