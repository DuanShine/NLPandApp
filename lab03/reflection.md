# LAB 03 — Reflection: Evolutionary NLP & Theoretical Mastery

## 1. Bảng so sánh tiến hóa các thế hệ biểu diễn ngôn ngữ (Mục 27)

| Representation | Context-dependent? | Sparse / Dense | Một từ có nhiều vector? | Chiều không gian ($d$) | Cơ chế trích xuất tri thức |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF** | Không | Sparse | Không | $\vert{}V\vert{}$ | Thống kê tần suất xuất hiện của từ trong văn bản và tần suất nghịch đảo trên toàn bộ tập dữ liệu (corpus). |
| **Co-occurrence Matrix** | Không | Sparse | Không | $\vert{}V\vert{}$ | Đếm tần suất đồng xuất hiện trực tiếp giữa các cặp từ trong cùng một cửa sổ ngữ cảnh cố định. |
| **Word2Vec (CBOW / Skip-gram)** | Không (Static) | Dense | Không (1 vector cố định / từ) | Cố định ($50, 100, 300, \dots$) | Mạng nơ-ron học dự đoán phân bố xác suất điều kiện cục bộ giữa từ mục tiêu và từ ngữ cảnh qua tối ưu hóa Gradient Descent (SGNS). |
| **Contextual Embedding (ELMo / BERT / Transformer)** | Có (Dynamic) | Dense | Có (mỗi ngữ cảnh sinh 1 vector riêng) | Cố định ($768, 1024, \dots$) | Cơ chế Self-Attention tính toán trọng số tương tác đa chiều giữa từ với toàn bộ câu văn ngay tại thời điểm suy luận. |

### Câu hỏi suy ngẫm: Tại sao từ "bank" cần contextual representation?

1. **Hiện tượng đa nghĩa trong ngôn ngữ tự nhiên:**
   * Từ "bank" là hiện thân tiêu biểu của tính đa nghĩa (Polysemy/Homonymy) với hai trường ngữ nghĩa hoàn toàn tách biệt:
     * *Ngữ nghĩa tài chính:* Tổ chức tín dụng, ngân hàng (*"I deposited money in the bank"*).
     * *Ngữ nghĩa địa lý:* Dải đất ven sông, bờ đê (*"We sat on the river bank watching the water"*).

2. **Rào cản của biểu diễn tĩnh (Static Embeddings như Word2Vec/GloVe):**
   * Các mô hình tĩnh ánh xạ mỗi token $w$ thành một vector cố định trong bảng tra cứu (lookup table). Khi từ "bank" xuất hiện ở cả hai miền ngữ cảnh khác nhau trong quá trình huấn luyện, thuật toán buộc phải cập nhật gradient về cùng một vector duy nhất.
   * Kết quả là vector của "bank" bị kéo về trạng thái trung bình dung hòa giữa hai miền nghĩa rời rạc, tạo ra một vector "lai tạp" lơ lửng ở giữa không gian tài chính và địa lý, không thể phản ánh chính xác bất kỳ sắc thái nghĩa thực tế nào.

3. **Sự bứt phá từ biểu diễn ngữ cảnh (Contextual Representation):**
   * Trong các kiến trúc dựa trên Transformer (BERT, RoBERTa), từ "bank" không chỉ dừng lại ở biểu diễn nhúng ban đầu mà tiếp tục đi qua các lớp Multi-Head Self-Attention.
   * Cơ chế Self-Attention tự động phân bổ trọng số dựa trên các từ xung quanh:
     * Nếu xuất hiện các từ ngữ cảnh như *"money"*, *"deposited"*, *"account"*, trọng số chú ý xoáy sâu vào các đặc trưng tài chính, định hướng vector $\vec{h}_{\text{bank}}$ về miền không gian kinh tế.
     * Nếu xuất hiện *"river"*, *"water"*, *"stream"*, trọng số chú ý sẽ điều hướng vector $\vec{h}_{\text{bank}}$ dịch chuyển thẳng sang không gian sinh thái địa lý.
   * Nhờ đó, mô hình khởi tạo được vô số biểu diễn vector động cho cùng một dạng từ vựng, giúp máy tính hiểu sâu sắc ý nghĩa ngữ cảnh thực sự.

---

## 2. Trả lời  random 1 trong 6 câu hỏi kiểm tra vấn đáp cá nhân (Individual Learning Check — Mục 29)

**Câu 1: Distributional Hypothesis là gì?**
* **Trả lời:** Giả thuyết phân bố do Zellig Harris đề xuất và J.R. Firth đúc kết qua câu *"You shall know a word by the company it keeps"*, khẳng định rằng các từ thường xuyên xuất hiện trong những ngữ cảnh phân bố giống nhau thì sẽ mang ý nghĩa ngữ nghĩa tương đồng nhau.
---

## 3. Tuyên bố sử dụng AI (AI Assistance Statement — Mục 28)

Tuân thủ nghiêm ngặt quy định liêm chính học thuật tại Mục 28 (AI Policy):

* **Công cụ hỗ trợ:** Google Antigravity Assistant (Gemini 3.1 Pro).
* **Mục đích:** Hỗ trợ gợi ý cấu trúc trình bày bảng so sánh, rà soát công thức định dạng Markdown và kiểm tra tính hợp lý của các hạng mục lý thuyết.
* **Nội dung do AI gợi ý:** Khung bảng đối chiếu đặc trưng các thế hệ biểu diễn ngôn ngữ và cấu trúc dàn ý tổng quan cho phần suy ngẫm.
* **Nội dung sinh viên thực hiện & tinh chỉnh:** Toàn bộ quá trình tính toán đại số trong bài tập Analogy, các tập huấn luyện cho câu *"the cat eats fish"*, bài phân tích định tính về nguyên nhân gây lỗi mô hình, cùng toàn bộ phần lập luận chi tiết cho 6 câu hỏi vấn đáp cá nhân đều do sinh viên tự nghiên cứu, biên soạn hoàn toàn bằng văn phong độc lập và đối chiếu thực nghiệm.
* **Xác minh:** Mọi kết quả số liệu trong `calculations.md`, file `results.csv` và ma trận ở `cooccurrence.py` đã được chạy thực tế, kiểm chứng bằng mã nguồn Python thuần và đảm bảo chính xác 100%.
