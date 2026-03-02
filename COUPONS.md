# Danh Sách Mã Giảm Giá

## 🎁 Các mã giảm giá hiện có:

### 1. **TOMATO10**
- Loại: Giảm giá %
- Giá trị: Giảm 10%
- Tối đa: 50,000đ
- Đơn tối thiểu: 0đ
- Mô tả: Giảm 10% cho đơn hàng - Tối đa 50k

### 2. **FREESHIP**
- Loại: Miễn phí vận chuyển
- Giá trị: 100%
- Tối đa: 30,000đ
- Đơn tối thiểu: 0đ
- Mô tả: Miễn phí vận chuyển

### 3. **WELCOME50**
- Loại: Giảm giá cố định
- Giá trị: 50,000đ
- Đơn tối thiểu: 200,000đ
- Mô tả: Giảm 50k cho đơn hàng từ 200k

### 4. **FLASH20**
- Loại: Giảm giá %
- Giá trị: Giảm 20%
- Tối đa: 100,000đ
- Đơn tối thiểu: 300,000đ
- Mô tả: Giảm 20% cho đơn từ 300k - Tối đa 100k

### 5. **VIP15**
- Loại: Giảm giá %
- Giá trị: Giảm 15%
- Tối đa: 200,000đ
- Đơn tối thiểu: 500,000đ
- Mô tả: Giảm 15% cho đơn từ 500k - Tối đa 200k

---

## 💡 Cách sử dụng:

1. Thêm sản phẩm vào giỏ hàng
2. Vào trang giỏ hàng (/cart)
3. Nhập mã giảm giá vào ô "Nhập mã giảm giá"
4. Click nút "Áp dụng"
5. Xem giảm giá được áp dụng ngay lập tức

## 🔧 Tùy chỉnh mã giảm giá:

Chỉnh sửa file `data/coupons.jsonl` để thêm/sửa/xóa mã:

```json
{
  "code": "NEWCODE",
  "discount_type": "percent",  // "percent", "fixed", "shipping"
  "discount_value": 15,
  "min_order": 100000,
  "max_discount": 50000,
  "expires": "2026-12-31",
  "usage_limit": 100,
  "used_count": 0,
  "description": "Mô tả mã giảm giá"
}
```

### Loại giảm giá:
- **percent**: Giảm % (cần set max_discount)
- **fixed**: Giảm số tiền cố định
- **shipping**: Miễn phí vận chuyển (discount_value = 100 nghĩa là 100%)

---

## 🎯 Test Cases:

1. **Test TOMATO10** với đơn 200k:
   - Giảm giá: 20k (10% của 200k)
   - Tổng sau giảm: 180k + ship

2. **Test WELCOME50** với đơn 150k:
   - ❌ Lỗi: "Đơn hàng tối thiểu 200,000đ"

3. **Test FREESHIP** với đơn 100k:
   - Giảm giá: 30k (free ship)
   - Tổng: 100k (không phí ship)

4. **Test VIP15** với đơn 600k:
   - Giảm giá: 90k (15% của 600k)
   - Miễn phí ship (đơn > 500k)
   - Tổng: 510k
