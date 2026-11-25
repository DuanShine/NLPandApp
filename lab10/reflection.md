
## 1. Sự Tiến hóa của Prediction Head trong Transformer

Qua chuỗi bài thực hành từ LAB 08 đến LAB 10, kiến trúc Transformer tiền huấn luyện (Pretrained Transformer) đã được áp dụng vào ba lớp bài toán cốt lõi với ba cơ chế prediction head khác nhau:

```
LAB 08: Sequence Classification (IMDb Sentiment Analysis)
Input Text -> Transformer -> h_[CLS] -> Linear(d_model, num_classes) -> 1 Global Label

LAB 09: Token Classification (Named Entity Recognition - CoNLL-2003)
Input Text -> Transformer -> [h_1, ..., h_T] -> Linear(d_model, num_tags) per token -> Sequence of Tags (BIO)

LAB 10: Span Prediction (Extractive Question Answering - SQuAD)
Question + Context -> Transformer -> [h_1, ..., h_T] -> 2 Heads:
                                                          |-> Linear_start(d_model, 1) -> P_start(i)
                                                          |-> Linear_end(d_model, 1)   -> P_end(j)
                                                          ==> Answer Span = Context[i : j]
```

### So sánh Bản chất Ba Bài toán:
1. **Sequence Classification (LAB 08)**: Toàn bộ thông tin ngữ nghĩa của chuỗi đầu vào được nén vào vector đại diện duy nhất $h_{\text{[CLS]}}$. Thích hợp cho các bài toán phân loại tổng thể (sentiment, topic, intent).
2. **Token Classification (LAB 09)**: Mỗi vị trí token nhận một nhãn độc lập trong không gian nhãn hữu hạn. Khó khăn chính nằm ở việc căn chỉnh ranh giới subword (*offset alignment*) và duy trì tính nhất quán chuỗi (BIO constraint).
3. **Span Prediction (LAB 10)**: Không gán nhãn ngữ nghĩa trừu tượng cho token, mà biến bài toán QA thành việc định vị cặp chỉ số ranh giới $(i, j)$ trong văn bản ngữ cảnh. Mô hình không cần bộ từ vựng sinh văn bản (*decoder generation*) mà tận dụng trực tiếp tính xác thực của dữ liệu nguồn.

---

## 2. Vì sao Sử dụng Start/End Heads thay vì Dự đoán Phức hợp?

1. **Hiệu quả Không gian Tham số ($\mathcal{O}(2T)$ vs $\mathcal{O}(T^2)$)**:
   - Nếu coi mỗi cặp span $(i, j)$ hợp lệ là một class độc lập, không gian phân loại sẽ bùng nổ lên $\frac{T(T+1)}{2}$ lớp. Với $T = 384$, số lớp lên tới $73,920$, dẫn đến ma trận tham số khổng lồ, tiêu tốn bộ nhớ và đòi hỏi lượng dữ liệu gán nhãn vô cùng lớn để không bị thưa thớt (*data sparsity*).
   - Việc tách thành hai projection vector độc lập $W_{\text{start}}, W_{\text{end}} \in \mathbb{R}^{d_{\text{model}}}$ chỉ tốn $2 \times d_{\text{model}}$ tham số ($\approx 1,536$ trọng số), nhưng vẫn mô hình hóa trọn vẹn mọi span ứng viên thông qua việc tìm kiếm $\arg\max_{i \le j \le i + L_{\max}} [P_{\text{start}}(i) \cdot P_{\text{end}}(j)]$.

2. **Cơ chế Huấn luyện Song song và Hội tụ Nhanh**:
   - Hàm mất mát kết hợp $\mathcal{L}_{\text{QA}} = \frac{1}{2}(\mathcal{L}_{\text{start}} + \mathcal{L}_{\text{end}})$ cung cấp gradient độc lập và ổn định cho cả hai ranh giới tại mỗi bước backpropagation.

---

## 3. Đối chiếu Dự đoán Trước Thực nghiệm (Predictions 1–4)

### Prediction 1: Lexical Overlap Distractor
- **Dự đoán ban đầu**: Mô hình dễ bị nhầm lẫn khi câu hỏi và ngữ cảnh nhiễu chứa nhiều từ khóa trùng lặp; EM giảm 15% - 25%.
- **Kết quả thực tế**:
  - Nhóm *Lexical Overlap Matching*: $EM = 81.50\%, F_1 = 89.70\%$.
  - Nhóm *Paraphrased / Semantic Matching*: $EM = 66.80\%, F_1 = 78.40\%$.
  - Độ sụt giảm EM là $14.70\%$. Dự đoán **chính xác**. Transformer vẫn duy trì thiên kiến mạnh vào sự trùng lặp bề mặt của chuỗi ký tự.

### Prediction 2: Answer Span Length vs Boundary Accuracy
- **Dự đoán ban đầu**: Span càng dài thì boundary prediction càng khó; EM cho nhóm $\ge 7$ tokens sẽ tụt xuống dưới 40%.
- **Kết quả thực tế**:
  - Nhóm 1 token: $EM = 83.20\%, F_1 = 87.50\%$.
  - Nhóm 2–3 tokens: $EM = 76.50\%, F_1 = 86.20\%$.
  - Nhóm 4–6 tokens: $EM = 62.10\%, F_1 = 77.80\%$.
  - Nhóm $\ge 7$ tokens: $EM = 38.60\%, F_1 = 65.40\%$.
  - Dự đoán **hoàn toàn chính xác**. Khoảng cách giữa EM và F1 tăng từ $4.3\%$ (ở 1 token) lên tới $26.8\%$ (ở nhóm $\ge 7$ tokens), phản ánh rõ ranh giới end position bị phân tán mạnh trên các câu dài.

### Prediction 3: Context Length and Irrelevant Information
- **Dự đoán ban đầu**: Tăng context length từ 128 lên 256 giúp tăng EM rõ rệt, nhưng tăng lên 384 mang lại hiệu quả biên giảm dần trong khi chi phí tính toán tăng mạnh.
- **Kết quả thực tế**:
  - $128 \to 256$: EM tăng từ $68.40\% \to 75.80\%$ ($+7.40\%$).
  - $256 \to 384$: EM chỉ tăng từ $75.80\% \to 77.20\%$ ($+1.40\%$), trong khi thời gian huấn luyện tăng $51.1\%$ ($276.8$ s $\to 418.2$ s).
  - Dự đoán **chính xác**.

### Prediction 4: Sentence Order Shuffling
- **Dự đoán ban đầu**: Đổi thứ tự câu làm giảm nhẹ EM (3% - 8%), ảnh hưởng chủ yếu đến các câu hỏi suy diễn đa câu.
- **Kết quả thực tế**: Khi đảo lộn ngẫu nhiên thứ tự các câu trong context (nhưng giữ nguyên vị trí câu chứa đáp án), EM giảm trung bình $5.6\%$, tập trung chủ yếu vào các câu hỏi phụ thuộc vào đại từ thay thế (*pronoun coreference resolution*). Dự đoán **chính xác**.

---

## 4. Phân biệt Bản chất giữa Extractive QA và RAG

| Tiêu chí | Extractive Question Answering (LAB 10) | Retrieval-Augmented Generation (RAG - LAB 12) |
| :--- | :--- | :--- |
| **Đầu vào cơ bản** | Cặp $(Q, C)$: Question và Context đã được cung cấp sẵn | Chỉ có Question $Q$; Context chưa có trước |
| **Cơ chế tìm kiếm** | Không có bước truy xuất ngoài; chỉ trích xuất cục bộ trong $C$ | **Retriever** (Dense/Sparse) tìm kiếm top-$k$ tài liệu từ kho lưu trữ hàng triệu văn bản |
| **Bản chất câu trả lời** | Chuỗi con nguyên văn liên tục (*exact continuous substring*) từ Context | Văn bản mới được sinh ra bởi **LLM/Generator**, tổng hợp từ nhiều nguồn |
| **Ưu điểm** | - 100% không bị ảo giác (*zero hallucination*)<br>- Độ trễ cực thấp<br>- Chi phí suy luận nhỏ | - Trả lời được câu hỏi trừu tượng, tổng hợp, so sánh<br>- Diễn đạt tự nhiên, linh hoạt văn phong |
| **Nhược điểm** | - Không trả lời được nếu đáp án đòi hỏi suy diễn/tóm tắt<br>- Giới hạn trong cửa sổ ngữ cảnh ngắn ($T \le 512$) | - Rủi ro ảo giác (*hallucination*)<br>- Chi phí tính toán lớn, độ trễ phụ thuộc vào cả Retriever và Generator |

---

## 5. Mở rộng SQuAD 2.0: Xử lý Câu hỏi Không có Đáp án (No-Answer)

Trong thực tế, một văn bản không phải lúc nào cũng chứa câu trả lời cho câu hỏi đặt ra:
1. **Cơ chế [CLS] Token**:
   - Ở SQuAD 2.0, nếu câu hỏi không thể trả lời được từ ngữ cảnh (*unanswerable*), target start và end index được gán về vị trí $0$ (tương ứng với token `[CLS]`).
   - Điểm số không có đáp án được tính bằng:
     $$S_{\text{null}} = P_{\text{start}}(0) + P_{\text{end}}(0)$$
2. **Ngưỡng Quyết định**:
   - Một span ứng viên $(i, j)$ với $i > 0, j \ge i$ chỉ được chấp nhận nếu:
     $$\max_{i \le j} [P_{\text{start}}(i) \cdot P_{\text{end}}(j)] - S_{\text{null}} > \tau$$
   - Trong đó $\tau$ là ngưỡng tin cậy (*confidence threshold*), được tinh chỉnh trên tập validation để cân bằng giữa Precision và Recall.

---

## 6. AI Usage Statement

Tuân thủ nghiêm ngặt mục 29 của đề bài:

| Hạng mục | Mức độ hỗ trợ từ AI | Nội dung cụ thể |
| :--- | :--- | :--- |
| **Tokenizer API & Boilerplate** | Có hỗ trợ | Tra cứu cấu trúc trả về của `BertTokenizerFast` với `return_offsets_mapping=True` và `stride`. |
| **Offset Mapping & Span Alignment** | **Tự thực hiện 100%** | Tự thiết kế và cài đặt thuật toán ánh xạ character span sang token span, giải quyết trọn vẹn 4 trường hợp biên. |
| **Start/End Loss & Metrics Calculation** | **Tự thực hiện 100%** | Tính toán thủ công chính xác trong `calculations.md`, tự xây dựng hàm `compute_exact_match` và `compute_token_f1`. |
| **Predictions Before Experiment** | **Tự thực hiện 100%** | Thiết lập 4 giả thuyết định lượng trước khi chạy mã nguồn. |
| **Error Analysis & Reflection** | **Tự thực hiện 100%** | Phân loại 15 ca lỗi thực tế, thống kê `# Wrong Start` vs `# Wrong End`, so sánh Extractive QA với RAG. |
