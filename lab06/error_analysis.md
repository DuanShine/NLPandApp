# LAB 06 — Báo cáo Phân tích Lỗi Định tính (Error Analysis): Attention Patterns & Behavior


---

## 1. Bảng tổng hợp các trường hợp phân tích lỗi

| Mã mẫu | Phân loại | Câu văn bản đầu vào (Review Text) | Nhãn thực tế (True) | Dự đoán (Pred) | Xác suất (Prob) | Các token nhận Attention cao nhất | Nhóm nguyên nhân chính |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **C1** | Đúng | *"The movie was surprisingly good despite a very slow and dull beginning."* | Positive | **Positive** | $0.884$ | *"good"* ($0.38$), *"surprisingly"* ($0.24$) | Nhận diện tương phản & Từ khóa cảm xúc |
| **C2** | Đúng | *"I did not like this movie at all; the plot was completely incoherent."* | Negative | **Negative** | $0.932$ | *"not"* ($0.31$), *"incoherent"* ($0.29$), *"like"* ($0.15$) | Cấu trúc phủ định cục bộ (Local Negation) |
| **C3** | Đúng | *"The actors were brilliant, and despite minor script flaws, the ending was stunning."* | Positive | **Positive** | $0.915$ | *"brilliant"* ($0.34$), *"stunning"* ($0.31$) | Phụ thuộc tầm xa trực tiếp qua Self-Attention |
| **E1** | **Sai** | *"I expected the movie to be a brilliant masterpiece, but it turned out to be an absolute disaster."* | Negative | **Positive** | $0.628$ | *"brilliant"* ($0.36$), *"masterpiece"* ($0.31$), *"disaster"* ($0.12$) | Cấu trúc tương phản giả định quá khứ (Expectation vs Reality) |
| **E2** | **Sai** | *"Oh, what a brilliant idea to have two hours of people staring at a blank wall in silence."* | Negative | **Positive** | $0.785$ | *"brilliant"* ($0.44$), *"idea"* ($0.18$) | Mỉa mai / Châm biếm (Sarcasm / Irony) |
| **E3** | **Sai** | *"The cinematography is breathtaking, though the screenplay is utterly shallow and tedious."* | Negative | **Positive** | $0.542$ | *"breathtaking"* ($0.41$), *"cinematography"* ($0.22$) | Cảm xúc đa chiều / Mơ hồ (Ambiguous Sentiment) |
| **E4** | **Sai** | *"Not once in the entire runtime did the director fail to impress the audience."* | Positive | **Negative** | $0.691$ | *"not"* ($0.35$), *"fail"* ($0.32$), *"impress"* ($0.11$) | Phủ định kép phức tạp (Double Negation) |

---

## 2. Phân tích định tính các trường hợp phân loại đúng (Correct Cases)

### Ca C1: Xử lý cấu trúc nhượng bộ tương phản
- **Câu văn:** *"The movie was surprisingly good despite a very slow and dull beginning."*
- **Hành vi Attention:**
  - Mô hình dồn trọng số tập trung vào cặp từ *"good"* ($\alpha = 0.38$) và phó từ biến đổi mức độ *"surprisingly"* ($\alpha = 0.24$).
  - Các từ tiêu cực ở vế nhượng bộ (*"slow", "dull"*) chỉ nhận trọng số nhỏ ($\alpha < 0.08$).
- **Lý giải:** Nhờ cơ chế Self-Attention tính toán độ tương đồng giữa tất cả các cặp từ, từ *"surprisingly"* đã tạo liên kết mạnh mẽ với *"good"*, làm nổi bật sắc thái khen ngợi bất ngờ và chi phối hoàn toàn Context Vector hướng về nhãn Positive.

### Ca C2: Nắm bắt cấu trúc phủ định trực tiếp (*"did not like"*)
- **Câu văn:** *"I did not like this movie at all; the plot was completely incoherent."*
- **Hành vi Attention:**
  - Token *"not"* nhận trọng số $\alpha = 0.31$, đồng thời có điểm tương tác cao với *"like"* ($\alpha = 0.15$). Token *"incoherent"* ở vế sau nhận $\alpha = 0.29$.
- **Lý giải:** Khác với mô hình túi từ (BoW) bị nhiễu do từ tích cực *"like"*, Self-Attention đã cho phép token *"like"* truy vấn trực tiếp đến *"not"*, tạo ra biểu diễn ngữ cảnh đảo chiều tiêu cực, giúp phân loại chính xác nhãn Negative với độ tự tin lên tới $93.2\%$.

### Ca C3: Phụ thuộc tầm xa vượt qua các mệnh đề chêm xen
- **Câu văn:** *"The actors were brilliant, and despite minor script flaws, the ending was stunning."*
- **Hành vi Attention:**
  - Hai từ khóa quyết định cảm xúc nằm ở hai đầu câu: *"brilliant"* (đầu câu) và *"stunning"* (cuối câu) nhận lần lượt $34\%$ và $31\%$ trọng số attention. Mệnh đề nhượng bộ ở giữa (*"despite minor script flaws"*) bị suy giảm trọng số.
- **Lý giải:** Trong khi Simple RNN thường làm suy giảm thông tin của *"brilliant"* do bị vế nhượng bộ chèn ép, Self-Attention kết nối trực tiếp hai đầu với khoảng cách đường truyền $\mathcal{O}(1)$, giữ nguyên vẹn cả hai đỉnh cảm xúc tích cực.

---

## 3. Phân tích chuyên sâu các trường hợp dự đoán sai (Incorrect Cases)

### Ca E1: Thất bại trước cấu trúc "Kỳ vọng đối lập Thực tế" (Expectation vs Reality)
- **Câu văn:** *"I expected the movie to be a brilliant masterpiece, but it turned out to be an absolute disaster."*
- **Nhãn thực tế:** Negative | **Mô hình dự đoán:** Positive ($P = 0.628$)
- **Khảo sát Attention Patterns:**
  - Token nhận Attention cao nhất: *"brilliant"* ($\alpha = 0.36$), *"masterpiece"* ($\alpha = 0.31$).
  - Token biểu thị cảm xúc thực tế: *"disaster"* chỉ nhận $\alpha = 0.12$; liên từ đối lập *"but"* nhận $\alpha = 0.05$.
- **Thiết lập và kiểm chứng giả thuyết (Hypothesis Testing):**
  - *Giả thuyết:* Mô hình sai không phải vì Self-Attention "nhìn nhầm từ", mà vì trong không gian Embedding học được, các từ mang cực tính cảm xúc cực mạnh như *"brilliant"* và *"masterpiece"* có độ lớn chuẩn vector (L2 norm) rất cao và kích hoạt giá trị tương đồng lớn với các token xung quanh.
  - Ngược lại, động từ nhận thức ở thì quá khứ *"expected"* không đủ sức làm triệt tiêu ảnh hưởng của hai danh từ tích cực đứng sau nó nếu không có cơ chế phân tích cú pháp phân cấp (hierarchical syntax).
  - *Kết luận:* Mô hình bị áp đảo bởi mật độ từ tích cực mạnh trong mệnh đề kỳ vọng ban đầu, dẫn đến việc Context Vector bị kéo lệch sang nhãn Positive.

### Ca E2: Giới hạn trước câu Châm biếm / Mỉa mai (Sarcasm / Irony)
- **Câu văn:** *"Oh, what a brilliant idea to have two hours of people staring at a blank wall in silence."*
- **Nhãn thực tế:** Negative | **Mô hình dự đoán:** Positive ($P = 0.785$)
- **Khảo sát Attention Patterns:**
  - Token nhận Attention vượt trội: *"brilliant"* ($\alpha = 0.44$), *"idea"* ($\alpha = 0.18$).
- **Thiết lập và kiểm chứng giả thuyết (Hypothesis Testing):**
  - *Giả thuyết:* Câu châm biếm sử dụng hình thức từ ngữ tích cực trên bề mặt chữ (*"brilliant idea"*) để diễn đạt sự chê bai gay gắt trong thực tế.
  - Mô hình Self-Attention chỉ dựa trên tương quan từ vựng cục bộ không thể phát hiện ra tính chất phi lý của hành động *"staring at a blank wall in silence"* để lật ngược ngữ nghĩa của *"brilliant"*.
  - *Kết luận:* Đây là giới hạn về tri thức thế giới (World Knowledge) và ngữ dụng học (Pragmatics), không thể giải quyết đơn thuần bằng việc tăng trọng số Attention.

### Ca E3: Cảm xúc đa chiều giữa Kỹ thuật và Nội dung (Domain-specific Ambiguity)
- **Câu văn:** *"The cinematography is breathtaking, though the screenplay is utterly shallow and tedious."*
- **Nhãn thực tế:** Negative | **Mô hình dự đoán:** Positive ($P = 0.542$)
- **Khảo sát Attention Patterns:**
  - Token nhận Attention cao nhất: *"breathtaking"* ($\alpha = 0.41$), *"cinematography"* ($\alpha = 0.22$).
  - Token tiêu cực: *"shallow"* ($\alpha = 0.14$), *"tedious"* ($\alpha = 0.11$).
- **Thiết lập và kiểm chứng giả thuyết (Hypothesis Testing):**
  - *Giả thuyết:* Trong các bài phê bình điện ảnh, người xem thường khen hình ảnh nhưng chê kịch bản. Ở đây, trọng số của từ *"breathtaking"* lớn hơn tổng hai từ *"shallow"* và *"tedious"*.
  - Liên từ *"though"* không đủ trọng số để báo hiệu cho bộ phân loại tuyến tính rằng vế sau liên từ mang trọng số quyết định cảm xúc chung của toàn bài.
  - *Kết luận:* Trọng số Attention phản ánh độ tương đồng biểu diễn, nhưng phép cộng trọng số (Weighted Sum) làm phẳng các mối quan hệ logic thứ bậc giữa các khía cạnh (Aspect-level sentiment).

### Ca E4: Thất bại trước cấu trúc Phủ định kép (Double Negation)
- **Câu văn:** *"Not once in the entire runtime did the director fail to impress the audience."*
- **Nhãn thực tế:** Positive | **Mô hình dự đoán:** Negative ($P = 0.691$)
- **Khảo sát Attention Patterns:**
  - Token nhận Attention cao nhất: *"not"* ($\alpha = 0.35$), *"fail"* ($\alpha = 0.32$).
- **Thiết lập và kiểm chứng giả thuyết (Hypothesis Testing):**
  - *Giả thuyết:* Cấu trúc đảo ngữ phủ định kép *"Not once ... did ... fail"* mang ý nghĩa khẳng định mạnh mẽ (*"always impressed"*).
  - Tuy nhiên, trong mô hình Self-Attention 1 lớp (Single-layer), hai token tiêu cực *"not"* và *"fail"* chú ý mạnh vào nhau và cùng kích hoạt các đặc trưng tiêu cực trong không gian ẩn, hoàn toàn áp đảo động từ tích cực *"impress"*.
  - *Kết luận:* Mô hình cần các tầng biểu diễn sâu hơn (Deep Multi-layer Transformer) để tổng hợp ý nghĩa phủ định kép thành một đặc trưng khẳng định bậc cao.

---

## 4. Tổng kết: Attention Patterns có đồng nghĩa với Diễn giải suy luận? (Attention is NOT Explanation)

1. **Phân biệt tương quan và nhân quả:**
   - Việc Attention tập trung vào một token chỉ chứng minh token đó có tích vô hướng $q^T k$ lớn trong không gian vector ẩn.
   - Nó **không chứng minh** rằng token đó là nguyên nhân duy nhất khiến mô hình đưa ra quyết định phân loại, bởi vì sau lớp Attention còn có ma trận chiếu giá trị $W_V$, lớp tuyến tính Classifier $W_{cls}$ và các hàm kích hoạt phi tuyến.
2. **Vai trò thực tế của Attention Heatmap:**
   - Visualization là công cụ chẩn đoán (Diagnostic tool) vô cùng hữu ích để kiểm tra xem mô hình có bị phân tâm vào các từ rác (noise/stopwords) hay không.
   - Tuy nhiên, không bao giờ được quy chụp: *"Mô hình nhìn vào từ X nên chắc chắn hiểu câu theo nghĩa Y"*. Cần luôn kiểm chứng giả thuyết bằng cách quan sát đồng thời cả phân phối Attention lẫn xác suất đầu ra (Output Logits).
