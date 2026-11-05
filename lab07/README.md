# LAB 07 — Transformer Block: Multi-Head Attention, Positional Encoding & Sequence Classification

---

## 1. Tổng quan Lab

Trong các bài thực hành trước, chúng ta đã chứng kiến từng bước chuyển dịch mang tính nền tảng của các phương pháp biểu diễn ngôn ngữ:
- **LAB 01 & LAB 04:** Biểu diễn túi từ thưa (TF-IDF) và véc-tơ trung bình tĩnh (Average Word2Vec) — hoàn toàn làm mất trật tự từ và ngữ cảnh động.
- **LAB 05:** Mạng hồi quy (Simple RNN, LSTM, GRU) — ghi nhận trật tự thời gian nhưng bị ràng buộc bởi nút cổ chai thông tin (Information Bottleneck) tại trạng thái ẩn cuối $\vec{h}_T$ và không thể tính toán song song trên GPU.
- **LAB 06:** Cơ chế Attention & Self-Attention — thiết lập đường truyền tương tác trực tiếp $\mathcal{O}(1)$ giữa mọi cặp token nhưng lại thiếu vắng khái niệm thứ tự từ (Permutation Invariant) và thiếu khả năng biến đổi phi tuyến cục bộ.

Trong **LAB 07**, chúng ta chính thức hoàn thiện kiến trúc nền tảng của kỷ nguyên AI đương đại: **Khối Transformer Encoder Block (Vaswani et al., 2017)**. Bằng cách tích hợp hài hòa 4 trụ cột công nghệ:
1. **Positional Encoding (Mã hóa vị trí):** Tiêm thông tin thứ tự thời gian vào mô hình không phụ thuộc thứ tự.
2. **Multi-Head Self-Attention (Cơ chế chú ý đa đầu):** Phân chia không gian biểu diễn thành nhiều lát cắt trực giao song song.
3. **Residual Connections & Layer Normalization (Add & Norm):** Duy trì đường cao tốc gradient và ổn định phương sai biểu diễn.
4. **Position-wise Feed-Forward Network (FFN):** Biến đổi phi tuyến sâu độc lập tại từng vị trí token.

Quy trình nghiên cứu và thực nghiệm trong Lab 07 gồm:
> Tính toán giải tích một Attention Head, Multi-Head Concatenation, Positional Encoding, FFN, Residual Gradient và Shape Tensors → Thiết lập 6 giả thuyết khoa học định lượng → Tự xây dựng khối Transformer Encoder từ đầu không dùng API có sẵn → Kiểm chứng thực nghiệm tính bất biến theo hoán vị trên hai câu *"dog bites man"* vs *"man bites dog"* → Khảo sát thực nghiệm số lượng Attention Heads ($h \in \{1, 2, 4, 8\}$) → Trực quan hóa tính đa dạng của 4 Attention Heads → So sánh đối chuẩn toàn diện với LSTM Baseline → Phân tích sự bùng nổ chi phí tính toán $\mathcal{O}(T^2)$ → So sánh chiến lược gộp đặc trưng Mean vs CLS Pooling → Ứng dụng phân loại cảm xúc nhận biết ngữ cảnh trên 5 mẫu câu → Phân tích lỗi định tính và chuẩn bị tiền đề cho BERT.

---

## 2. Mục tiêu học tập cốt lõi (Learning Outcomes)

1. **LO1 — Transformer Paradigm:** Nắm vững cấu trúc tổng thể và luồng thông tin của Transformer Encoder Block (Multi-Head Attention $\to$ Add & Norm $\to$ FFN $\to$ Add & Norm).
2. **LO2 — Mathematical Grounding:** Tự tính toán giải tích chính xác các bước chiếu $Q, K, V$, Scaled Attention Matrix, phép ghép nối Concatenation, Positional Encoding và Feed-Forward Network.
3. **LO3 — Permutation Invariance & Positional Encoding:** Giải thích nguyên lý toán học và kiểm chứng thực nghiệm tại sao Self-Attention cần Positional Encoding để phân biệt trật tự chuỗi (*"dog bites man"* vs *"man bites dog"*).
4. **LO4 — Multi-Head Subspace Diversity:** Hiểu rõ tại sao Multi-Head Attention phân tách biểu diễn thành $h$ không gian con ($d_k = d_{model}/h$) để nắm bắt đồng thời các mối quan hệ ngữ pháp, ngữ nghĩa và điểm neo dấu câu khác nhau.
5. **LO5 — Optimization via Residual & LayerNorm:** Phân tích vai trò của số hạng đồng nhất $I$ trong đạo hàm $\frac{\partial (x + F(x))}{\partial x}$ giúp triệt tiêu hiện tượng tiêu biến gradient khi xếp chồng mạng sâu.
6. **LO6 — Modular Architecture Implementation:** Tự xây dựng mô hình Transformer hoàn chỉnh từ các module cơ sở (không sử dụng `TransformerEncoderLayer`), tuân thủ nghiêm ngặt quy chuẩn không comment trong code.
7. **LO7 — Quadratic Scaling Trade-off:** Đo lường và đánh giá chi phí tính toán bậc hai $\mathcal{O}(T^2)$ khi tăng độ dài chuỗi từ $32 \to 256$, hiểu rõ lý do Attention bị giới hạn chiều dài ngữ cảnh trên GPU.
8. **LO8 — Pooling Strategy:** So sánh định lượng và lý giải bản chất giữa hai chiến lược trích xuất vector chuỗi: Mean Pooling toàn cục và First-token / CLS-style Pooling.
9. **LO9 — Empirical Comparison:** So sánh đối chuẩn hiệu năng phân loại cảm xúc (Accuracy, Precision, Recall, $F_1$, thời gian huấn luyện) giữa Transformer Encoder và đường cơ sở LSTM.
10. **LO10 — Error Diagnosis & Attention Limits:** Phân tích định tính các ca lỗi do câu châm biếm (sarcasm) hoặc cấu trúc giả định quá khứ (expectation vs reality), phân biệt rõ ranh giới giữa mẫu hình Attention số học và lập luận nhân quả (Attention is NOT Explanation).

---

## 3. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `lab07/` bao gồm đầy đủ **13 tệp tin** theo đúng quy định tại **Mục 32. Deliverables**:

```
w7/
├── README.md                           # Báo cáo tổng quan, kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.pdf                    # Lời giải ngắn gọn, chuẩn xác 7 bài tập giải tích (Attention head, Multi-head, PE, FFN, Residual, Shapes, Complexity)
├── prediction.pdf                      # Thiết lập 6 giả thuyết nghiên cứu định lượng kèm lý giải cơ chế và mức độ tin cậy
├── transformer_block.py                # Module PyTorch tự cài đặt Multi-Head Attention, FFN, TransformerBlock và Classifier (không comment)
├── positional_encoding.py              # Module cài đặt Sinusoidal Positional Encoding thuần NumPy và PyTorch (không comment)
├── transformer_classification.ipynb    # Notebook thực nghiệm hoàn chỉnh từ Mục 11 đến Mục 26 (code sạch, không comment, markdown học thuật)
├── results.csv                         # Bảng số liệu thực nghiệm định lượng chi tiết của các cấu hình mô hình
├── positional_encoding.png             # Đồ thị Heatmap và các đường sóng tần số của Positional Encoding
├── attention_heads.png                 # Đồ thị ma trận nhiệt trực quan hóa tính đa dạng mẫu hình của 4 Attention Heads
├── learning_curves.png                 # Đồ thị hàm mất mát Cross-Entropy và Validation Accuracy qua 10 epochs (Transformer vs LSTM)
├── confusion_matrix.png                # Ma trận nhầm lẫn của Transformer Encoder Classifier trên tập kiểm thử IMDb
├── error_analysis.md                   # Bảng phân tích định tính chuyên sâu 6 trường hợp đúng/sai và kiểm chứng giả thuyết
└── reflection.md                       # Tiến hóa từ LAB 01-07, đối chuẩn 6 giả thuyết, giải đáp 9 câu hỏi vấn đáp và AI Policy
```

---
