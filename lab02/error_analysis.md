## Phân tích các trường hợp dự đoán sai

| Category | Input Context | Prediction | Target Word | Confidence |
|---|---|---|---|---:|
| Incorrect 1 | `climate change` | `.` | `impacts` | 0.2476 |
| Incorrect 2 | `the company` | `’` | `has` | 0.1447 |

### Trường hợp 1: Dự đoán dấu câu thay vì từ nội dung

- **Ngữ cảnh đầu vào:** `climate change`
- **Từ mô hình dự đoán:** `.`
- **Từ kỳ vọng:** `impacts`
- **Độ tự tin:** `0.2476`

**Phân tích:**  
Cụm `climate change` có thể xuất hiện ở nhiều vị trí khác nhau trong văn bản, bao gồm cả vị trí gần cuối câu. Do dấu câu được giữ lại như một token riêng trong quá trình tiền xử lý, dấu chấm có thể xuất hiện với tần suất tương đối cao sau cụm từ này. Mô hình N-gram chỉ dựa trên tần suất xuất hiện của các từ trong ngữ cảnh ngắn nên không thể xác định được ý nghĩa của toàn bộ câu. Vì vậy, mô hình lựa chọn dấu chấm thay vì từ `impacts`.

### Trường hợp 2: Dự đoán dấu nháy thay vì động từ

- **Ngữ cảnh đầu vào:** `the company`
- **Từ mô hình dự đoán:** `’`
- **Từ kỳ vọng:** `has`
- **Độ tự tin:** `0.1447`

**Phân tích:**  
Lỗi này liên quan đến cách dấu nháy được xử lý trong quá trình tách từ. Trong văn bản tiếng Anh, dấu nháy có thể xuất hiện trong nhiều cấu trúc khác nhau, đặc biệt là các dạng sở hữu cách. Do đó, mô hình có thể học được mối liên hệ giữa danh từ và dấu nháy từ dữ liệu huấn luyện. Với ngữ cảnh chỉ gồm hai từ `the company`, Bigram không có đủ thông tin để phân biệt các cấu trúc có thể xuất hiện phía sau. Vì vậy, mô hình lựa chọn dấu nháy thay vì động từ `has`.
