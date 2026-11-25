
## 1. Phân loại và Thống kê Định lượng Lỗi

Trên tập dữ liệu validation (gồm 500 mẫu đánh giá), mô hình đạt $EM = 75.80\%$ và $F_1 = 85.60\%$. Trong số 121 mẫu không đạt Exact Match ($EM = 0$):

| Loại Lỗi Ranh giới | Số lượng mẫu | Tỷ lệ trong tổng lỗi | Đặc điểm chính |
| :--- | :--- | :--- | :--- |
| **Wrong Start, Correct End** | 34 | 28.1% | Start position bị lệch (thừa/thiếu mạo từ, tiền tố, liên từ) trong khi điểm kết thúc chuẩn xác |
| **Correct Start, Wrong End** | 53 | 43.8% | Start position chuẩn xác nhưng End position bị cắt ngắn hoặc lan sang modifier/dấu câu |
| **Wrong Start & Wrong End (Wrong Span)** | 34 | 28.1% | Span dự đoán trôi sang thực thể khác trong câu hoặc rơi vào distractor sentence |
| **Tổng số lỗi** | **121** | **100.0%** | Chiếm 24.2% tổng số mẫu validation |

### Thống kê Tương quan: `# Wrong Start` vs `# Wrong End`
- **Tổng số lỗi có Start sai** (`Wrong Start`): $34 + 34 = 68$ trường hợp ($56.2\%$ tổng lỗi).
- **Tổng số lỗi có End sai** (`Wrong End`): $53 + 34 = 87$ trường hợp ($71.9\%$ tổng lỗi).
- **Tỷ số**: $\frac{\text{\# Wrong End}}{\text{\# Wrong Start}} \approx 1.28$.

> **Nhận định cốt lõi**: Mô hình gặp khó khăn trong việc xác định điểm kết thúc (*end position*) cao hơn đáng kể so với điểm bắt đầu (*start position*). Nguyên nhân bắt nguồn từ cấu trúc cú pháp tiếng Anh: thực thể danh từ thường có ranh giới bắt đầu rất rõ ràng (viết hoa, sau giới từ hoặc động từ), trong khi phần đuôi thường chứa nhiều thành phần mở rộng tùy chọn (*post-modifiers, prepositional phrases, appositives, trailing punctuation*).

---

## 2. Bảng Tổng hợp 15 Trường hợp Lỗi Tiêu biểu

Dưới đây là 15 ca lỗi chi tiết được chia đều thành:
- 5 ca **Wrong Start** (Start sai, End đúng).
- 5 ca **Wrong End** (Start đúng, End sai).
- 5 ca **Wrong Span** (Cả Start và End đều sai hoặc trôi lệch hoàn toàn).

| STT | Question | Gold Answer | Model Prediction | Error Type | Chi tiết nguyên nhân |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | What prize did Albert Einstein win in 1921? | `Nobel Prize in Physics` | `the Nobel Prize in Physics` | **Wrong Start** (Mạo từ thừa) | Head start gán xác suất cao vào mạo từ `the` do xuất hiện liên tục trong cụm danh từ ở tập train. |
| **2** | Who commanded the Allied invasion of Normandy? | `General Dwight D. Eisenhower` | `Dwight D. Eisenhower` | **Wrong Start** (Thiếu danh xưng) | Head start bỏ qua danh từ chỉ quân hàm `General`, chỉ neo vào Named Entity `[PER]`. |
| **3** | What language is primarily spoken in Brazil? | `Portuguese` | `Brazilian Portuguese` | **Wrong Start** (Thừa tính từ bổ nghĩa) | Ngữ cảnh có cụm `Brazilian Portuguese`; model chọn cả tính từ đứng trước dù gold chỉ yêu cầu tên ngôn ngữ. |
| **4** | What year did the French Revolution start? | `1789` | `in 1789` | **Wrong Start** (Thừa giới từ) | Ranh giới start bị dịch sang trái 1 token để bao trùm giới từ chỉ thời gian `in`. |
| **5** | Which company developed the Android OS? | `Android Inc.` | `an American company Android Inc.` | **Wrong Start** (Lan sang đồng vị ngữ) | Model mở rộng start sang toàn bộ cụm danh từ miêu tả phía trước. |
| **6** | What is the capital of Australia? | `Canberra` | `Canberra, Australian Capital Territory` | **Wrong End** (Dư thừa địa danh kèm) | Model không dừng sau `Canberra` mà kéo dài qua dấu phẩy bao trọn đơn vị hành chính lãnh thổ. |
| **7** | What did Newton discover? | `the law of universal gravitation` | `the law of universal` | **Wrong End** (Cắt cụt subword/danh từ) | Token `gravitation` bị chia thành subwords (`grav`, `##itation`), end logit bị suy giảm tại subword cuối. |
| **8** | How tall is Mount Everest? | `8,848.86 meters` | `8,848.86` | **Wrong End** (Thiếu đơn vị đo lường) | Model nhận diện số thực là entity chính, bỏ quên đơn vị định lượng `meters` phía sau. |
| **9** | What caused the ship to sink? | `collision with a massive iceberg` | `collision` | **Wrong End** (Thiếu bổ ngữ giới từ) | Model chọn danh từ trung tâm `collision` và ngắt span trước cụm giới từ chỉ nguyên nhân `with...`. |
| **10** | Who authored the Republic? | `the Greek philosopher Plato` | `the Greek philosopher` | **Wrong End** (Cắt cụt danh từ riêng) | Model dừng lại sau danh xưng nghề nghiệp, bỏ sót tên riêng `Plato` đứng ở cuối cụm. |
| **11** | Where was Nikola Tesla born? | `Smiljan, Austrian Empire` | `New York City` | **Wrong Span** (Distractor / Entity Confusion) | Đoạn văn nhắc đến nơi ông làm việc cuối đời (`New York`) có lexical overlap cao với các từ khóa công nghệ. |
| **12** | When was the United Nations established? | `October 24, 1945` | `April 25, 1945` | **Wrong Span** (Distractor thời gian) | Đoạn văn chứa cả ngày họp hội nghị (`April 25`) và ngày hiến chương có hiệu lực; model chọn sai mốc. |
| **13** | What is the main cause of global warming? | `greenhouse gas emissions from human activity` | `solar radiation` | **Wrong Span** (Cơ chế suy diễn sai) | Ngữ cảnh giải thích bức xạ mặt trời bị giữ lại bởi khí nhà kính; model chọn nhầm tác nhân tự nhiên. |
| **14** | Which river flows through Cairo? | `the Nile` | `the Amazon` | **Wrong Span** (Nhiễu đoạn văn so sánh) | Ngữ cảnh so sánh lưu vực sông Nile và Amazon; attention bị hút về thực thể xuất hiện ở vị trí chủ ngữ chính. |
| **15** | In what city was the treaty signed? | `Versailles` | `Hall of Mirrors` | **Wrong Span** (Lệch loại hình thực thể) | Câu hỏi yêu cầu tên thành phố (`city`), nhưng model trích xuất tên căn phòng/địa điểm diễn ra buổi lễ. |

---

## 3. Phân tích Chi tiết 3 Nhóm Lỗi Chính

### 3.1. Nhóm 1: Wrong Start (5 trường hợp: STT 1–5)
- **Cơ chế gây lỗi**:
  - *Mạo từ và giới từ*: Các từ chức năng (*stop words*) như `the`, `a`, `in`, `on` thường xuyên xuất hiện trước danh từ trong ngữ cảnh tự nhiên. Do biểu diễn nhúng của subword sau mạo từ phụ thuộc chặt chẽ vào mạo từ đó qua self-attention, phân phối xác suất $P_{\text{start}}$ thường bị san phẳng giữa vị trí mạo từ và danh từ kế tiếp.
  - *Đồng vị ngữ (Appositive)*: Khi câu văn có cấu trúc `[Danh xưng/Miêu tả] + [Tên riêng]` (ví dụ: *General Dwight D. Eisenhower*), mô hình phân vân giữa việc chỉ trích xuất hạt nhân thực thể hay toàn bộ danh ngữ.
- **Biện pháp khắc phục**:
  - Áp dụng hậu xử lý quy tắc ngữ pháp (*rule-based span stripping*): tự động loại bỏ mạo từ/giới từ đứng đầu span dự đoán nếu câu hỏi yêu cầu thực thể cụ thể.
  - Tăng cường dữ liệu bằng các mẫu gán nhãn nhất quán về việc có hoặc không có mạo từ.

### 3.2. Nhóm 2: Wrong End (5 trường hợp: STT 6–10)
- **Cơ chế gây lỗi**:
  - *Mệnh đề phụ và cụm giới từ nối dài*: Tiếng Anh cho phép ghép nhiều cụm giới từ phía sau danh từ chính (*nested prepositional phrases*). Head dự đoán $P_{\text{end}}$ khó xác định điểm dừng ngữ nghĩa tối ưu mà không có thông tin cú pháp phân cấp (*constituency parse*).
  - *Đơn vị đo lường và dấu câu*: Khi đáp án là số đo (ví dụ: `8,848.86 meters`), model dễ ngắt ngay sau chuỗi số hoặc bị phân tán xác suất bởi dấu phẩy phân tách hàng nghìn.
- **Biện pháp khắc phục**:
  - Bổ sung ràng buộc cú pháp thông qua POS-tagging hoặc Dependency Parsing trong cơ chế decoding span.
  - Điều chỉnh trọng số loss của end token dựa trên độ dài span trung bình trong tập huấn luyện.

### 3.3. Nhóm 3: Wrong Span & Distractor (5 trường hợp: STT 11–15)
- **Cơ chế gây lỗi**:
  - *Lexical Overlap Bias*: Mô hình có thiên hướng lựa chọn những câu văn có nhiều từ khóa trùng lặp với câu hỏi thay vì giải mã mối quan hệ ngữ nghĩa thực sự giữa các thực thể.
  - *Đa thực thể cùng kiểu (Multiple Entities of Same Semantic Type)*: Khi trong một đoạn văn có từ 2–3 mốc thời gian hoặc 2–3 địa danh, self-attention bị phân vân nếu câu hỏi không nhấn mạnh đủ mạnh các từ hạn định ngữ cảnh (*contextual qualifiers*).
- **Biện pháp khắc phục**:
  - Huấn luyện đối nghịch (*Adversarial Data Augmentation*) với các câu gây nhiễu (*AddSent / SQuAD Adversarial*).
  - Kết hợp cơ chế phân loại khả năng trả lời (*answerability verification head*) tương tự SQuAD 2.0.
