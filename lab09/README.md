# LAB 09 — Named Entity Recognition and Information Extraction

---

## 1. Tổng quan Lab

LAB 08 đã khép lại chuỗi **Sequence Classification**: văn bản đi qua tokenizer và pretrained Transformer, sau đó chỉ biểu diễn `[CLS]` được đưa vào classification head để dự đoán **một nhãn cho cả chuỗi**:
$$X = (x_1, \dots, x_T) \to y$$

Trong **LAB 09**, chúng ta giữ nguyên backbone pretrained nhưng thay đổi đầu ra sang **Token Classification**: mỗi token nhận một nhãn riêng:
$$X = (x_1, \dots, x_T) \to Y = (y_1, \dots, y_T)$$

Bài toán cụ thể là **Named Entity Recognition (NER)** trên **CoNLL-2003** với 4 loại thực thể `PER`, `ORG`, `LOC`, `MISC` theo lược đồ **BIO** (`B-X`, `I-X`, `O`). Classification head tuyến tính được áp dụng lên **mọi vị trí token** của hidden states:
$$H \in \mathbb{R}^{B \times T \times d} \xrightarrow{\;Z = HW + b\;} Z \in \mathbb{R}^{B \times T \times C}$$

Vấn đề kỹ thuật trung tâm là **label alignment**: dataset NER gán nhãn theo *word*, trong khi Transformer hoạt động trên *subword token IDs*. Lab này sử dụng **Strategy A (First-subword)** làm chiến lược chính — chỉ subword đầu tiên của mỗi từ mang nhãn thật, các subword sau và special tokens nhận `-100` (IGNORE) nên không đóng góp vào loss:
$$\mathcal{L} = -\sum_{i \,:\, y_i \neq \text{IGNORE}} \log P(y_i \mid x)$$
và đối chiếu với **Strategy B (Label propagation `B-X → I-X`)**.

Quy trình nghiên cứu và thực nghiệm trong Lab 09 gồm:
> Tính tay BIO tagging (Exercise 1), entity boundary "University of California Berkeley" (Exercise 2) và tensor shape $H, Z, Y$ (Exercise 3) → Thiết lập 3 dự đoán trước thí nghiệm → Cài đặt label alignment với `word_ids()` (Strategy A vs B) và kiểm chứng `ignore_index=-100` → Baseline Token features + Logistic Regression → Pretrained BERT token classifier → Exp 1: Tokenization analysis trên 20 câu (tỉ lệ $R = \#\text{subword tokens} / \#\text{words}$) → Exp 2: Baseline vs Pretrained Transformer → Exp 3: Token-level vs Entity-level evaluation → Exp 4: Phân tích theo loại thực thể PER / ORG / LOC / MISC → Exp 5: Phân tích theo độ dài thực thể 1 / 2 / 3 / 4+ tokens → Exp 6: Error analysis (FP / FN / boundary errors) → Context analysis (cùng surface form, khác entity type) → Mini Information Extraction trên 30 câu tin tức → Reflection và cầu nối sang LAB 10 (Extractive Question Answering).

---

## 2. Mục tiêu học tập cốt lõi (Learning Outcomes)

1. **LO1 — Sequence vs Token Classification:** Phân biệt bài toán một nhãn cho cả chuỗi ($X \to y$) với một nhãn cho từng token ($X \to Y$).
2. **LO2 — Named Entity Recognition:** Giải thích bài toán NER và các loại thực thể `PER`, `ORG`, `LOC`, `MISC` trong CoNLL-2003.
3. **LO3 — BIO Tagging:** Hiểu vai trò của `B-X`, `I-X`, `O` trong việc biểu diễn cả **loại** lẫn **ranh giới** thực thể.
4. **LO4 — Annotation → Token Labels:** Chuyển annotation thực thể thành chuỗi nhãn token-level.
5. **LO5 — Label Alignment:** Hiểu vấn đề căn chỉnh giữa word, subword token và label; áp dụng nhất quán Strategy A (first-subword + `-100`) và phân tích ảnh hưởng so với Strategy B.
6. **LO6 — Transformer Token Classification:** Sử dụng pretrained Transformer với classification head trên mọi token, theo dõi đầy đủ `tokens`, `input_ids`, `attention_mask`, `word_ids`, `labels`, `logits`, `predictions`.
7. **LO7 — Fine-tuning trên CoNLL-2003:** Huấn luyện và đánh giá mô hình NER trên cùng test set với baseline.
8. **LO8 — Precision / Recall / F1 cho NER:** Tính và giải thích các chỉ số với TP, FP, FN xác định ở mức thực thể.
9. **LO9 — Token-level vs Entity-level Metrics:** Nhận ra vì sao token accuracy cao không đồng nghĩa với trích xuất thực thể đúng (exact span match).
10. **LO10 — Error Analysis:** Phân tích lỗi NER theo các nhóm wrong type, wrong boundary, missing entity, spurious entity.
11. **LO11 — Nguồn gốc lỗi:** Nhận diện lỗi do tokenization, context, entity boundary, rare entity và domain shift.
12. **LO12 — Information Extraction:** Sử dụng NER như bước đầu để biến văn bản phi cấu trúc thành dữ liệu có cấu trúc (`PERSON`, `ORG`, `LOCATION`).

---

## 3. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `lab09/` bao gồm đầy đủ các tệp tin theo đúng quy định tại **Mục 29. Deliverables**:

```
lab09/
├── README.md                     # Báo cáo tổng quan, mục tiêu học tập và cấu trúc bài nộp
├── calculations.md               # Lời giải tay: BIO "Elon Musk founded SpaceX in California." (Ex 1), entity boundary "University of California Berkeley" (Ex 2), shape H / Z / Y (Ex 3)
├── prediction.md                 # 3 dự đoán trước thí nghiệm (Transformer vs word-feature baseline, entity dài vs ngắn, entity hiếm)
├── baseline_ner.py               # Baseline: token features + Logistic Regression, extract_entities, compute_entity_metrics, compute_prf
├── tokenizer_alignment.ipynb     # Label alignment Strategy A (-100) vs Strategy B, kiểm chứng ignore_index, Exp 1: Tokenization analysis 20 câu
├── ner_finetuning.ipynb          # BERT token classifier, shape check, Exp 2–5, context analysis, mini Information Extraction
├── results.csv                   # Bảng tổng hợp: Baseline vs BERT (Strategy A / B) và Mini IE 30 câu — Token Acc, Entity P / R / F1
├── entity_metrics.csv            # Entity-level metrics theo loại thực thể (PER / ORG / LOC / MISC) và theo độ dài (1 / 2 / 3 / 4+ tokens)
├── entity_length_analysis.png    # Biểu đồ Entity F1 theo độ dài thực thể (Baseline LR vs Pretrained BERT)
├── confusion_matrix.png          # Ma trận nhầm lẫn nhãn BIO của Pretrained BERT trên CoNLL-2003 test
├── error_analysis.md             # Bảng 5 FP / 5 FN / 5 boundary errors, nguyên nhân cốt lõi và hướng khắc phục
├── information_extraction.py     # Pipeline Raw text → NER → Entity extraction → JSON, đánh giá entity-level trên 30 câu tin tức có gold
└── reflection.md                 # Reflection, kiểm chứng 3 dự đoán, giới hạn của BIO phẳng và AI Usage Statement
```
