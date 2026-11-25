# LAB 10 — Extractive Question Answering

---

## 1. Tổng quan Lab

LAB 08 và LAB 09 đã sử dụng cùng một pretrained Transformer nhưng thay đổi prediction head:
- **LAB 08 (Sequence Classification):** Biểu diễn `[CLS]` đi qua classification head để dự đoán **một nhãn cho cả chuỗi**: $X \to y$.
- **LAB 09 (Token Classification / NER):** Classification head được áp dụng lên **mọi token** để dự đoán một nhãn BIO cho từng vị trí: $X \to Y = (y_1, \dots, y_T)$.

Trong **LAB 10**, prediction head tiếp tục thay đổi sang **Span Prediction**. Model nhận cặp câu hỏi và ngữ cảnh, sau đó dự đoán vị trí bắt đầu và kết thúc của đáp án nằm sẵn trong context:
$$(Q, C) \to (\text{start}, \text{end}), \qquad A = C_{i:j}, \quad i \le j$$

Bài toán cụ thể là **Extractive Question Answering** trên **SQuAD** (Stanford Question Answering Dataset). Input có dạng `[CLS] question [SEP] context [SEP]`; hidden states được đưa qua hai classifier tuyến tính độc lập để sinh phân phối $P_{\text{start}}(i)$ và $P_{\text{end}}(j)$ trên các token. Hàm mất mát là trung bình của hai cross-entropy:
$$\mathcal{L} = \frac{\mathcal{L}_{\text{start}} + \mathcal{L}_{\text{end}}}{2}, \qquad \mathcal{L}_{\text{start}} = -\log P_{\text{start}}(i), \quad \mathcal{L}_{\text{end}} = -\log P_{\text{end}}(j)$$

Vấn đề kỹ thuật trung tâm là **offset mapping**: SQuAD cung cấp `answer_start` theo *character index*, trong khi Transformer làm việc trên *token index*. Lab này dùng `BertTokenizerFast` với `return_offsets_mapping=True` để ánh xạ character span của đáp án sang `(start_token, end_token)`, kết hợp `truncation="only_second"` và **sliding window** (`stride`) để xử lý context dài hơn `max_length` mà không đánh mất đáp án ở ranh giới cắt.

Quy trình nghiên cứu và thực nghiệm trong Lab 10 gồm:
> Tính tay start / end cho "Paris" và "capital of France" (Exercise 1), start loss $-\log(0.60)$ (Exercise 2) và character span → token span của "Paris" (Exercise 3) → Thiết lập 4 dự đoán trước thí nghiệm → Khảo sát `input_ids`, `attention_mask`, `token_type_ids`, `offset_mapping`, `overflow_to_sample_mapping` → Cài đặt answer span alignment và kiểm tra các trường hợp biên (subword, truncation, ngoài window) → Baseline keyword overlap → Pretrained BERT với start / end classifier → Exp 1: Context length 128 / 256 / 384 → Exp 2: Fine-tuning và learning curves (train / val loss, EM, F1) → Exp 3: EM vs F1 và các ví dụ EM = 0 nhưng F1 > 0 → Exp 4: Lexical overlap vs paraphrased question → Exp 5: Answer length 1 / 2–3 / 4–6 / 7+ tokens → Error analysis (wrong start / wrong end / wrong span) → Mini Document QA → Phân biệt Extractive QA và RAG → Reflection và cầu nối sang LAB 11 (Semantic Retrieval).

---

## 2. Mục tiêu học tập cốt lõi (Learning Outcomes)

1. **LO1 — Extractive QA:** Giải thích bài toán trích xuất đáp án từ context có sẵn, không sinh văn bản mới.
2. **LO2 — Ba loại Prediction Head:** Phân biệt classification (1 nhãn / chuỗi), token classification (1 nhãn / token) và span prediction (start + end).
3. **LO3 — QA as Span Prediction:** Biểu diễn QA dưới dạng $(\text{question}, \text{context}) \to (\text{start}, \text{end})$.
4. **LO4 — QA Tokenization:** Hiểu cách ghép question và context thành `[CLS] question [SEP] context [SEP]`, vai trò của `token_type_ids` và không giả định mọi Transformer đều dùng chúng.
5. **LO5 — Offset Mapping:** Xử lý `offset_mapping` và phân biệt character index, word index, token index.
6. **LO6 — Character Span → Token Span:** Ánh xạ `answer_start` / `answer_end` sang `start_token` / `end_token`, kể cả khi đáp án bắt đầu hoặc kết thúc giữa subword, bị truncation hoặc nằm ngoài window.
7. **LO7 — Start / End Loss:** Tính $\mathcal{L}_{\text{start}}$, $\mathcal{L}_{\text{end}}$ và giải thích vì sao cần hai prediction độc lập.
8. **LO8 — Fine-tuning cho QA:** Fine-tune pretrained Transformer với start / end classifier và theo dõi learning curves.
9. **LO9 — Exact Match & F1:** Đánh giá bằng EM (khớp hoàn toàn sau normalization) và token-level F1 (mức độ overlap).
10. **LO10 — Error Analysis:** Phân tích lỗi wrong start, wrong end, boundary, no answer và distractor context; thống kê #wrong start so với #wrong end.
11. **LO11 — Hạn chế của Extractive QA:** Nhận diện các trường hợp không thể trả lời: đáp án không có trong context, context bị cắt, nhiều span hợp lý.
12. **LO12 — Chuẩn bị cho Retrieval & RAG:** Phân biệt QA với context có sẵn và hệ thống cần retriever để tìm context (LAB 11–12).

---

## 3. Cấu trúc thư mục nộp bài (Deliverables)

Thư mục `lab10/` bao gồm đầy đủ các tệp tin theo đúng quy định tại **Mục 31. Deliverables**:

```
lab10/
├── README.md                     # Báo cáo tổng quan, mục tiêu học tập và cấu trúc bài nộp
├── calculations.md               # Lời giải tay: start / end (Ex 1), start loss -log(0.60) (Ex 2), character span → token span của "Paris" (Ex 3)
├── prediction.md                 # 4 dự đoán trước thí nghiệm (lexical distractor, độ dài answer, context nhiễu, đảo thứ tự câu)
├── tokenizer_alignment.ipynb     # BertTokenizerFast + offset_mapping, align_answer_to_tokens, 4 edge cases, sliding window với stride
├── qa_model.py                   # BertForSpanQA (start / end heads), compute_qa_loss, extract_answer_span, normalize_answer, EM, token F1
├── qa_finetuning.ipynb           # Exp 1–5: context length, learning curves, EM vs F1, lexical vs paraphrase, answer length, Document QA demo
├── results.csv                   # Bảng tổng hợp: Baseline vs BERT (max_len 128 / 256 / 384), matching, answer length — EM, F1, train time
├── learning_curves.png           # Train / validation loss và validation EM / F1 qua 5 epochs
├── em_f1_analysis.png            # So sánh EM và F1 giữa Baseline và các cấu hình BERT
├── answer_length_analysis.png    # EM / F1 theo nhóm độ dài đáp án (1 / 2–3 / 4–6 / 7+ tokens)
├── error_analysis.md             # Thống kê wrong start vs wrong end, bảng 15 ca lỗi (5 wrong start / 5 wrong end / 5 wrong span)
├── document_qa.py                # DocumentQASystem: Question + Document → sliding window → QA model → answer span + confidence
└── reflection.md                 # Reflection LAB 08–10, kiểm chứng 4 dự đoán, QA vs RAG, SQuAD 2.0 no-answer và AI Usage Statement
```
