# Hướng dẫn sử dụng Gemini API cho Chatbot

## Đã hoàn thành ✅

Hệ thống chatbot đã được cập nhật để sử dụng **Google Gemini AI** thay vì file Excel (.xlsx).

### Những gì đã thay đổi:

1. **Xóa bỏ phụ thuộc cũ:**
   - ❌ Không còn dùng `pandas`, `openpyxl`, `rapidfuzz`
   - ❌ Không còn cần file `tomato_answer_question.xlsx`
   - ❌ Không còn fuzzy matching

2. **Thêm mới:**
   - ✅ Thư viện `google-generativeai` 
   - ✅ Hàm `get_gemini_response()` gọi API Gemini
   - ✅ Chatbot thông minh hơn với AI tự nhiên

## API Key đã được cấu hình

File `.env` của bạn đã có sẵn:
```env
GEMINI_API_KEY=AIzaSyDehXDN7AH94iKHMCgJj3ODcIQwfl4PUwo
```

## Cách chạy

```bash
# Kích hoạt virtual environment
.\venv\Scripts\activate

# Chạy ứng dụng
python tomato/app.py
```

Hoặc:

```bash
cd tomato
python app.py
```

## Cách lấy API Key mới (nếu cần)

1. Truy cập: https://aistudio.google.com/app/apikey
2. Đăng nhập bằng tài khoản Google
3. Nhấn **"Create API Key"**
4. Copy API key và thay thế trong file `.env`:
   ```env
   GEMINI_API_KEY=your-new-api-key-here
   ```

## Tính năng chatbot

Chatbot bây giờ có thể:
- Trả lời câu hỏi về cà chua một cách tự nhiên
- Hiểu ngữ cảnh tốt hơn
- Đưa ra câu trả lời chi tiết hơn
- Tự động từ chối câu hỏi không liên quan đến cà chua
- Hỗ trợ tiếng Việt tự nhiên

## Ví dụ câu hỏi test

- "Cà chua có những loại nào?"
- "Cách trồng cà chua như thế nào?"
- "Bệnh héo xanh trên cà chua là gì?"
- "Cà chua có chứa vitamin gì?"
- "Mùa nào trồng cà chua tốt nhất?"

## Giới hạn API

- **Gemini 1.5 Flash**: Miễn phí với giới hạn 15 requests/phút
- Đủ để sử dụng cho dự án học tập và demo

## Troubleshooting

### Lỗi: "Hệ thống chatbot chưa được cấu hình"
➡️ Kiểm tra file `.env` có `GEMINI_API_KEY` chưa

### Lỗi: "API key not valid"
➡️ API key hết hạn hoặc không đúng, tạo key mới

### Chatbot không trả lời
➡️ Kiểm tra kết nối internet và log file trong `logs/app.log`

## Logs

Tất cả các câu hỏi và trả lời được ghi vào:
- `data/chat_logs.jsonl`

Xem chi tiết lỗi trong:
- `logs/app.log`
- `logs/error.log`

## So sánh: Trước và Sau

| Tính năng | Trước (Excel) | Sau (Gemini AI) |
|-----------|---------------|-----------------|
| Nguồn dữ liệu | File .xlsx cố định | AI tự động sinh |
| Độ linh hoạt | Chỉ trả lời câu match | Hiểu ngữ cảnh |
| Cập nhật | Phải edit file Excel | Tự động |
| Độ thông minh | Fuzzy matching đơn giản | AI tự nhiên |
| Ngôn ngữ | Giới hạn | Tự nhiên, mượt mà |

---

**Lưu ý**: API key trong file `.env` hiện tại là của bạn. Hãy giữ bí mật và không commit lên Git!
