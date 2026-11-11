# LAB 08 — Pretrained Transformers and BERT Fine-Tuning

---

## 1. Tổng quan Lab

Trong các bài thực hành trước, chúng ta đã đi từ cơ chế Attention đến một Transformer hoàn chỉnh:
- **LAB 06 (Attention & Self-Attention):** Thay trạng thái nén $\vec{h}_T$ của RNN bằng Context Vector $\vec{c} = \sum_i \alpha_i \vec{v}_i$, cho phép mọi cặp token tương tác trực tiếp với chi phí $\mathcal{O}(T^2)$.
- **LAB 07 (Transformer from scratch):** Lắp ráp Multi-Head Attention, Positional Encoding, Feed-Forward, Residual + LayerNorm thành Transformer Encoder và huấn luyện từ trọng số ngẫu nhiên:
  $$D_{\text{task}} \to \text{Transformer} \to \text{task loss}$$

Một Transformer huấn luyện từ đầu phải học đồng thời **biểu diễn ngôn ngữ** và **nhiệm vụ phân loại** chỉ từ dữ liệu của chính bài toán. Khi dữ liệu downstream nhỏ, đây là điểm nghẽn lớn.

Trong **LAB 08**, chúng ta chuyển sang mô hình **Pretraining → Fine-tuning**. Một Transformer được huấn luyện trước trên kho văn bản lớn bằng Masked Language Modeling:
$$L_{\text{MLM}} = -\sum_{i \in M} \log P(y_i \mid x)$$
sau đó được tiếp tục huấn luyện trên tập dữ liệu cụ thể:
$$\theta_{\text{pretrained}} \to \theta_{\text{task}}$$
Biểu diễn `[CLS]` của encoder pretrained được đưa qua một classification head tuyến tính (thành phần task-specific) để phân loại cảm xúc IMDb.

Quy trình nghiên cứu và thực nghiệm trong Lab 08 gồm:
> Tính tay MLM loss và phân tích tokenization → Thiết lập 3 dự đoán trước thí nghiệm → Khảo sát tokenizer, subword, special tokens và attention mask → Cài đặt pretrained encoder + classification head trên `[CLS]` → Exp 1: Pretrained vs Transformer from scratch → Exp 2: Frozen encoder (feature extraction) vs Fine-tuning → Exp 3: Data efficiency 10% / 25% / 50% / 100% → Exp 4: Max sequence length 64 / 128 / 256 → Exp 5: Domain shift IMDb → Tweets → Đánh giá Accuracy / Precision / Recall / F1 + Confusion Matrix → Confidence analysis → Error analysis → Reflection và cầu nối sang LAB 09 (Token Classification / NER).

---

## 2. Mục tiêu học tập cốt lõi (Learning Outcomes)

1. **LO1 — Pretraining vs Fine-tuning:** Phân biệt giai đoạn học biểu diễn ngôn ngữ trên kho văn bản lớn và giai đoạn thích nghi với nhiệm vụ cụ thể; fine-tuning không bắt đầu từ trọng số ngẫu nhiên.
2. **LO2 — Masked Language Modeling:** Mô tả ý tưởng dự đoán token bị che $P(w_t \mid w_{\neq t})$ và tính cross-entropy loss trên tập vị trí bị mask.
3. **LO3 — Tokenizer & Subword:** Giải thích luồng raw text → tokenizer → token IDs → embedding → Transformer, và cách subword token xử lý từ hiếm / từ chưa gặp.
4. **LO4 — Special Tokens & Attention Mask:** Nhận diện vai trò của `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]` và cách attention mask loại padding khỏi tính toán.
5. **LO5 — [CLS] Classification:** Sử dụng biểu diễn `[CLS]` làm đại diện chuỗi cho classification head tuyến tính.
6. **LO6 — Training Regimes:** Phân biệt training from scratch, feature extraction (frozen encoder) và full fine-tuning.
7. **LO7 — Controlled Experiments:** So sánh pretrained vs from scratch, frozen vs fine-tuned, các mức dữ liệu huấn luyện và độ dài chuỗi với cùng split, metric và seed.
8. **LO8 — Evaluation beyond Accuracy:** Đánh giá bằng Accuracy, Precision, Recall, F1, Confusion Matrix và phân tích độ tin cậy $\text{confidence} \neq \text{correctness}$.
9. **LO9 — Limitations:** Nhận diện domain shift, giới hạn tokenizer, sequence length, chi phí tính toán $\mathcal{O}(T^2)$ và catastrophic forgetting.
10. **LO10 — Error Analysis:** Phân loại lỗi (negation, sarcasm, long-range dependency, ambiguous sentiment, ...) theo chuỗi observation → evidence → interpretation.

---

## 3. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `lab08/` bao gồm đầy đủ các tệp tin theo đúng quy định tại **Mục 25. Deliverables**:

```
lab08/
├── README.md                     # Báo cáo tổng quan, thiết lập, kết quả thực nghiệm và hướng dẫn chạy
├── calculations.md               # Lời giải tay: MLM loss (Exercise 1) và tokenization "The cats are playing." (Exercise 2)
├── prediction.md                 # 3 dự đoán trước thí nghiệm (pretrained vs scratch, frozen vs fine-tune, 10% data)
├── tokenization.ipynb            # Tokenizer DistilBERT: token sequence, token IDs, special tokens, attention mask, padding, truncation
├── pretrained_classification.py  # Module: load dữ liệu, pretrained encoder + head [CLS], Transformer from scratch, train/eval, vẽ hình
├── fine_tuning.ipynb             # Notebook chạy Exp 1–5, metrics, confusion matrix, confidence & error analysis, reproducibility
├── results.csv                   # Bảng số liệu định lượng của toàn bộ các lần chạy
├── data_efficiency.png           # Đường cong Accuracy / F1 theo % training data (pretrained vs from scratch)
├── confusion_matrix.png          # Ma trận nhầm lẫn của DistilBERT fine-tuned trên IMDb test
├── confidence_analysis.png       # Phân bố confidence và accuracy theo nhóm Low / Medium / High
├── error_analysis.md             # Bảng FP / FN / sai nhưng rất tự tin và các câu hỏi Q1–Q6
└── reflection.md                 # Reflection 300–500 từ và Individual Learning Check
```

