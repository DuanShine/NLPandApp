# LAB 06 — Attention Mechanism: Scaled Dot-Product & Self-Attention

---

## 1. Tổng quan Lab

Trong các bài thực hành trước, chúng ta đã tiếp cận hai trường phái biểu diễn ngôn ngữ lớn:
- **LAB 01 - 04 (Fixed-size Document Representations):** Mô hình túi từ thưa (TF-IDF) và véc-tơ ngữ nghĩa trung bình (Average Word2Vec). Các mô hình này nén văn bản thành một véc-tơ duy nhất và hoàn toàn làm mất trật tự từ (Word Order) cũng như ngữ cảnh động.
- **LAB 05 (Recurrent Sequence Modeling):** Xử lý văn bản tuần tự từ trái sang phải qua các mạng hồi quy (Simple RNN, LSTM, GRU) và lấy trạng thái ẩn cuối cùng $\vec{h}_T$ làm đại diện phân loại:
  $$\vec{x}_1 \to \vec{x}_2 \to \dots \to \vec{x}_T \implies \vec{h}_T \implies \text{Classifier}$$

Mặc dù mạng hồi quy ghi nhận được trật tự từ, việc nén toàn bộ câu văn dù dài hàng trăm từ vào một vector $\vec{h}_T$ có kích thước cố định đã tạo ra **Nút cổ chai thông tin (Information Bottleneck)** nghiêm trọng, đồng thời gây ra hiện tượng tiêu biến gradient (Vanishing Gradient) ở các token cách xa thời điểm cuối.

Trong **LAB 06**, chúng ta chính thức tiếp cận một trong những bước ngoặt vĩ đại nhất của lịch sử NLP: **Cơ chế Tập trung (Attention Mechanism)** và **Tự chú ý (Self-Attention)**. Thay vì phụ thuộc vào một trạng thái nén $\vec{h}_T$, mô hình duy trì toàn bộ các vector giá trị và cho phép bộ phân loại trích xuất linh hoạt một **Vector ngữ cảnh (Context Vector)**:
$$\vec{c} = \sum_{i=1}^T \alpha_i \vec{v}_i, \quad \text{với } \alpha = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$

Quy trình nghiên cứu và thực nghiệm trong Lab 06 gồm:
> Tính toán giải tích Attention Scores, Softmax Weights, Context Vector → Phân tích vai trò của Scaling $\sqrt{d_k}$ và Temperature $T$ → Chứng minh toán học tính ổn định số học (Numerically Stable Softmax) → Thiết lập 5 giả thuyết nghiên cứu định lượng → Tự cài đặt Scaled Dot-Product Attention và Self-Attention thuần NumPy → Khảo sát sự dịch chuyển Attention theo Query (Sentiment vs Topic) → Trực quan hóa ma trận liên kết ngữ nghĩa Self-Attention Heatmap → Đánh giá thực nghiệm so sánh LSTM vs Self-Attention Classifier → Phân tích bài toán chi phí tính toán $\mathcal{O}(T^2)$ ("Attention không miễn phí") → Xây dựng ứng dụng phân loại cảm xúc thời gian thực → Phân tích định tính 7 trường hợp lỗi và kiểm chứng giả thuyết "Attention is NOT Explanation" → Tổng kết bài học và chuẩn bị cho Transformer.

---

## 2. Mục tiêu học tập cốt lõi (Learning Outcomes)

1. **LO1 — Attention Paradigm:** Hiểu rõ hạn chế của nút cổ chai thông tin $\vec{h}_T$ trong mạng hồi quy và lý do chuyển dịch sang biểu diễn động bằng Context Vector $\vec{c} = \sum \alpha_i \vec{v}_i$.
2. **LO2 — Mathematical Computation:** Tính toán giải tích chính xác bằng tay qua từng bước: Tích vô hướng Dot-Product Scores $\to$ Chuẩn hóa xác suất Softmax Weights $\to$ Tổ hợp tuyến tính Context Vector.
3. **LO3 — Scaling Factor:** Lý giải bằng xác suất và đạo hàm tại sao phép chia $\sqrt{d_k}$ giúp kéo phương sai của điểm số về 1, ngăn chặn hàm Softmax rơi vào vùng bão hòa triệt tiêu gradient.
4. **LO4 — Temperature Control:** Phân tích ảnh hưởng của tham số nhiệt độ $T$ tới độ phân cực (Sharpness) và Entropy thông tin của phân phối Softmax.
5. **LO5 — Numerical Stability:** Cài đặt thuật toán Stable Softmax trừ $\max(x)$ và chứng minh toán học tính bảo toàn xác suất, loại trừ hoàn toàn nguy cơ tràn số dương (Overflow $\to \infty$ và `NaN`).
6. **LO6 — Self-Attention Formulation:** Nắm vững cấu trúc 3 không gian chiếu $Q = X W_Q, K = X W_K, V = X W_V$ để thiết lập đường truyền tương tác trực tiếp $\mathcal{O}(1)$ giữa mọi cặp token trong cùng một chuỗi.
7. **LO7 — Quadratic Complexity $\mathcal{O}(T^2)$:** Hiểu sâu sắc cái giá phải trả của Attention ("Attention không miễn phí"): số lượng phần tử và dung lượng bộ nhớ VRAM tăng theo cấp số nhân bậc hai theo chiều dài chuỗi, tạo ra giới hạn độ dài văn bản trên GPU.
8. **LO8 — Query-Dependent Flexibility:** Nhận diện tính chất động của Attention: cùng một câu văn sẽ sinh ra phân phối Attention hoàn toàn khác nhau tùy thuộc vào mục tiêu của Query.
9. **LO9 — Empirical Superiority:** So sánh định lượng hiệu năng và thời gian huấn luyện giữa LSTM và Self-Attention Classifier trên tập dữ liệu IMDb Movie Reviews.
10. **LO10 — Critical Error Analysis:** Phân tích định tính các mẫu hình Attention (Attention Patterns) trên các ca đúng/sai thực tế, hiểu rõ ranh giới giữa tương quan số học và lập luận nhân quả (Attention is NOT Explanation).

---

## 3. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `w6/` bao gồm đầy đủ **10 tệp tin** theo đúng quy định tại **Mục 32. Deliverables**:

```
w6/
├── README.md                 # Báo cáo tổng quan, kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.md           # Lời giải chi tiết 9 bài toán giải tích (Scores, Softmax, Context, Scaling, Temperature, Stable Softmax, Self-Attention, O(T^2))
├── prediction.md             # Thiết lập 5 giả thuyết nghiên cứu định lượng kèm lý giải cơ chế và mức độ tin cậy
├── attention_numpy.py        # Module thuần NumPy cài đặt Stable Softmax, Scaled Dot-Product Attention và Self-Attention (không comment)
├── self_attention.ipynb      # Notebook thực nghiệm hoàn chỉnh từ Mục 4 đến Mục 26 (code sạch, không comment, markdown học thuật)
├── results.csv               # Bảng số liệu thực nghiệm định lượng chi tiết của các mô hình
├── attention_heatmap.png     # Đồ thị trực quan hóa ma trận liên kết Self-Attention và biểu đồ dịch chuyển trọng số theo Query
├── confusion_matrix.png      # Ma trận nhầm lẫn của bộ phân loại Self-Attention Classifier trên tập kiểm thử IMDb
├── error_analysis.md         # Bảng phân tích định tính chuyên sâu 7 trường hợp đúng/sai và kiểm chứng giả thuyết Attention
└── reflection.md             # So sánh RNN vs Attention, phân tích chi phí O(T^2), giải đáp 9 câu hỏi vấn đáp và AI Policy
```
