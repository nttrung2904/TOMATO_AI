# 💳 Hướng dẫn Tích hợp Thanh toán VNPay và MoMo

## 📋 Tổng quan

Dự án đã được tích hợp 2 cổng thanh toán phổ biến tại Việt Nam:
- **VNPay**: Hỗ trợ thẻ ATM, Visa, Mastercard, JCB, QR Code
- **MoMo**: Ví điện tử MoMo

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình môi trường

Copy file `.env.example` sang `.env` và điền thông tin:

```bash
cp .env.example .env
```

Cập nhật các biến sau trong `.env`:

#### VNPay Configuration

```env
VNPAY_TMN_CODE=your-vnpay-tmn-code
VNPAY_SECRET_KEY=your-vnpay-secret-key
VNPAY_PAYMENT_URL=https://sandbox.vnpayment.vn/paymentv2/vpcpay.html
VNPAY_RETURN_URL=http://localhost:5000/payment/vnpay/callback
```

#### MoMo Configuration

```env
MOMO_PARTNER_CODE=your-momo-partner-code
MOMO_ACCESS_KEY=your-momo-access-key
MOMO_SECRET_KEY=your-momo-secret-key
MOMO_PAYMENT_URL=https://test-payment.momo.vn/v2/gateway/api/create
MOMO_RETURN_URL=http://localhost:5000/payment/momo/callback
MOMO_NOTIFY_URL=http://localhost:5000/payment/momo/ipn
```

## 🔑 Lấy API Credentials

### VNPay

#### Test (Sandbox)
1. Truy cập: https://sandbox.vnpayment.vn/
2. Đăng ký tài khoản test
3. Lấy thông tin trong Developer Portal:
   - **TMN Code**: Mã website của bạn
   - **Secret Key**: Mã bảo mật

#### Test Cards (VNPay Sandbox)
```
Ngân hàng: NCB
Số thẻ: 9704198526191432198
Tên chủ thẻ: NGUYEN VAN A
Ngày phát hành: 07/15
Mật khẩu OTP: 123456
```

#### Production
1. Đăng ký doanh nghiệp tại: https://vnpay.vn/
2. Hoàn thiện hồ sơ và ký hợp đồng
3. Nhận thông tin production từ VNPay
4. Cập nhật `VNPAY_PAYMENT_URL=https://pay.vnpay.vn/vpcpay/pay.html`

### MoMo

#### Test
1. Truy cập: https://developers.momo.vn/
2. Đăng ký tài khoản developer
3. Tạo app và lấy credentials:
   - **Partner Code**
   - **Access Key**
   - **Secret Key**

#### Test Account
```
Số điện thoại: 0963181714
OTP: Nhận từ app MoMo test
```

#### Production
1. Đăng ký doanh nghiệp tại: https://business.momo.vn/
2. Hoàn thiện hồ sơ kinh doanh
3. Nhận thông tin production
4. Cập nhật `MOMO_PAYMENT_URL=https://payment.momo.vn/v2/gateway/api/create`

## 🌐 Public URL cho Callback (Development)

Khi test trên localhost, payment gateway không thể gọi callback về. Dùng **ngrok** để expose local server:

### Cài đặt ngrok

```bash
# Download từ https://ngrok.com/download
# hoặc
choco install ngrok  # Windows
brew install ngrok   # Mac
```

### Chạy ngrok

```bash
ngrok http 5000
```

Ngrok sẽ cung cấp URL public, ví dụ: `https://abc123.ngrok.io`

### Cập nhật Callback URLs

Cập nhật trong `.env`:

```env
VNPAY_RETURN_URL=https://abc123.ngrok.io/payment/vnpay/callback
MOMO_RETURN_URL=https://abc123.ngrok.io/payment/momo/callback
MOMO_NOTIFY_URL=https://abc123.ngrok.io/payment/momo/ipn
```

## 📝 Quy trình thanh toán

### 1. User chọn sản phẩm và checkout
- Chọn phương thức thanh toán: COD / VNPay / MoMo
- Điền thông tin giao hàng

### 2. Tạo đơn hàng
- Server tạo order và lưu vào `data/orders.jsonl`
- Trả về `order_id`

### 3. Tạo giao dịch thanh toán (VNPay/MoMo)
- Gọi API `/api/payment/create` với order_id và amount
- Server tạo payment URL với signature
- Frontend redirect user đến payment gateway

### 4. User thanh toán
- User nhập thông tin thẻ/ví
- Xác thực và thanh toán

### 5. Callback
- Payment gateway gọi callback về server
- Server verify signature
- Cập nhật order status: `pending` → `paid`
- Hiển thị kết quả cho user

## 🔒 Bảo mật

### Signature Verification
- **VNPay**: HMAC SHA512
- **MoMo**: HMAC SHA256

Code tự động verify signature từ payment gateway để đảm bảo dữ liệu không bị giả mạo.

### Best Practices
1. ✅ Luôn verify signature trong callback
2. ✅ Không tin tưởng return URL (user có thể fake)
3. ✅ Chỉ cập nhật order status trong IPN/callback từ server payment gateway
4. ✅ Log tất cả transactions để audit
5. ✅ Không expose Secret Key trong client-side code
6. ✅ Sử dụng HTTPS trong production

## 🧪 Testing

### Test Flow

1. Khởi động app:
```bash
cd tomato
python app.py
```

2. (Optional) Khởi động ngrok:
```bash
ngrok http 5000
```

3. Truy cập: http://localhost:5000/shop

4. Thêm sản phẩm vào giỏ hàng

5. Checkout và chọn payment method:
   - **COD**: Hoàn tất ngay
   - **VNPay**: Redirect đến sandbox VNPay
   - **MoMo**: Redirect đến test MoMo

6. Thanh toán với thông tin test

7. Kiểm tra kết quả callback

### Test Cases

| Scenario | Expected Result |
|----------|-----------------|
| Thanh toán thành công | Order status = `paid`, hiển thị success page |
| Thanh toán thất bại | Order status = `pending`, hiển thị error page |
| Cancel payment | Redirect về cart, hiển thị lỗi |
| Invalid signature | Reject transaction, log warning |
| Duplicate callback | Idempotent, không xử lý lại |

## 📂 Files đã thêm/sửa

```
web_tomato/
├── tomato/
│   ├── app.py                    # ✏️ Thêm payment routes
│   └── payment.py                # ✨ NEW - Payment gateway integration
├── templates/
│   ├── cart.html                 # ✏️ Thêm VNPay/MoMo options
│   ├── payment_success.html     # ✨ NEW - Success page
│   └── payment_failed.html      # ✨ NEW - Failed page
├── .env.example                  # ✏️ Thêm payment configs
├── requirements.txt              # ✏️ Thêm requests
└── PAYMENT_SETUP.md             # ✨ NEW - This file
```

## 🐛 Troubleshooting

### Lỗi: "VNPay/MoMo not configured"
- Kiểm tra `.env` có đầy đủ credentials chưa
- Restart app sau khi cập nhật `.env`

### Callback không được gọi
- Kiểm tra URL có accessible từ internet không (dùng ngrok)
- Xem logs: `logs/app.log` và `logs/error.log`
- Kiểm tra firewall/antivirus

### Invalid signature
- Kiểm tra Secret Key đúng chưa
- Kiểm tra version API đúng chưa
- Xem logs để debug raw signature string

### Payment successful nhưng order chưa được cập nhật
- Kiểm tra IPN/callback có được gọi không
- Xem logs trong `logs/app.log`
- Verify signature thành công chưa

## 📞 Support

- **VNPay Support**: support@vnpay.vn
- **MoMo Support**: hotro@momo.vn
- **Project Issues**: [GitHub Issues](your-repo-url)

## 📚 Documentation

- [VNPay API Docs](https://sandbox.vnpayment.vn/apis/docs/)
- [MoMo API Docs](https://developers.momo.vn/v3/)

## ✅ Checklist trước khi deploy Production

- [ ] Đổi sang production URLs
- [ ] Cập nhật credentials production
- [ ] Enable HTTPS
- [ ] Set `SESSION_COOKIE_SECURE=True`
- [ ] Thay đổi `SECRET_KEY` mạnh
- [ ] Test kỹ payment flow
- [ ] Setup monitoring và alerts
- [ ] Backup database thường xuyên
- [ ] Log rotation và retention policy
- [ ] Security audit
- [ ] Load testing

---

**Chúc bạn triển khai thành công! 🎉**
