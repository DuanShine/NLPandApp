Category       Input Context       Prediction     Target Word    Confidence
--------------------------------------------------------------------------------
Correct 1      climate change      .              impacts        0.2476    
Correct 2      one of the          most           most           0.0196    
Incorrect 1    the company         ’              has            0.1447    
Incorrect 2    renewable energy    sources        sources        0.1231    
--------------------------------------------------------------------------------

#### 22.1. Nhóm các trường hợp dự đoán chính xác

**Trường hợp 1: Dự đoán trúng thuật ngữ có tính liên kết chặt chẽ**

- Ngữ cảnh đầu vào: `renewable energy`
- Từ mô hình dự đoán: `sources`
- Từ kỳ vọng: `sources`
- Độ tự tin: `0.1231`
- Phân tích nguyên nhân: Cụm từ năng lượng tái tạo đi kèm với chữ nguồn là một thuật ngữ chuẩn hóa xuất hiện với tần suất cao trong dữ liệu huấn luyện. Nhờ sự xuất hiện lặp lại của tổ hợp này, mô hình có nhiều cơ sở thống kê để ưu tiên từ `sources`. Điều này giúp hệ thống đưa ra dự đoán chính xác.

**Trường hợp 2: Dự đoán đúng khuôn mẫu ngữ pháp cố định**

- Ngữ cảnh đầu vào: `one of the`
- Từ mô hình dự đoán: `most`
- Từ kỳ vọng: `most`
- Độ tự tin: `0.0196`
- Phân tích nguyên nhân: Đây là một cấu trúc ngữ pháp phổ biến trong tiếng Anh. Khi gặp chuỗi `one of the`, mô hình có thể dựa vào tần suất xuất hiện trong dữ liệu để nhận biết các từ thường đứng tiếp theo. Trong trường hợp này, `most` có xác suất cao hơn các lựa chọn khác nên mô hình đưa ra dự đoán chính xác.

---

#### 22.2. Nhóm các trường hợp dự đoán sai

**Trường hợp 3: Sai do thói quen ngắt câu và tác dụng phụ của bước tiền xử lý**

- Ngữ cảnh đầu vào: `climate change`
- Từ mô hình dự đoán: `.`
- Từ kỳ vọng: `impacts`
- Độ tự tin: `0.2476`
- Phân tích nguyên nhân: Trong dữ liệu văn bản, cụm `climate change` có thể xuất hiện ở nhiều vị trí khác nhau, trong đó có những trường hợp đứng gần cuối câu. Do dấu câu được giữ lại như một token riêng, dấu chấm có thể xuất hiện với tần suất cao sau cụm từ này. Vì vậy, mô hình ưu tiên dấu chấm thay vì từ `impacts`.

**Trường hợp 4: Sai do ảnh hưởng của dạng sở hữu cách**

- Ngữ cảnh đầu vào: `the company`
- Từ mô hình dự đoán: `'`
- Từ kỳ vọng: `has`
- Độ tự tin: `0.1447`
- Phân tích nguyên nhân: Lỗi này xuất phát từ cách hệ thống xử lý dấu nháy trong quá trình tách từ. Trong dữ liệu tiếng Anh, dấu nháy có thể xuất hiện trong nhiều cấu trúc khác nhau, đặc biệt là các dạng sở hữu cách. Vì vậy, mô hình có thể học được mối liên hệ khá mạnh giữa danh từ và dấu nháy. Với ngữ cảnh chỉ gồm hai từ `the company`, thông tin chưa đủ để mô hình xác định chắc chắn rằng từ tiếp theo phải là một động từ như `has`, dẫn đến việc lựa chọn dấu nháy.