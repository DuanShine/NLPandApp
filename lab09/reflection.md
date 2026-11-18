# LAB 09 — Báo cáo Tổng kết & Suy ngẫm (Reflection): Từ Token Classification đến Information Extraction

## 1. So sánh Sequence Classification (LAB 08) vs. Token Classification (LAB 09)

| Tiêu chí | Sequence Classification (LAB 08) | Token Classification / NER (LAB 09) |
| :--- | :--- | :--- |
| **Ánh xạ bài toán** | $X = (x_1, \dots, x_T) \to y$ (1 nhãn / câu) | $X = (x_1, \dots, x_T) \to Y = (y_1, \dots, y_T)$ (1 nhãn / token) |
| **Vector đầu vào Classification Head** | Vector đại diện `[CLS]` ($H_{:,0,:} \in \mathbb{R}^{B \times d}$) | Toàn bộ trạng thái ẩn $H \in \mathbb{R}^{B \times T \times d}$ |
| **Shape Logits đầu ra** | $(B, C)$ | $(B, T, C)$ |
| **Vấn đề Subword Alignment** | Không cần căn chỉnh nhãn từng token | **Bắt buộc căn chỉnh** giữa word-level BIO labels và subword tokens |
| **Hàm mất mát (Loss)** | Cross-Entropy chuẩn trên câu | Cross-Entropy có `ignore_index=-100` cho special & non-first subwords |
| **Thước đo đánh giá cốt lõi** | Accuracy, Macro/Binary F1 | **Entity-level Exact Match Precision, Recall, F1** |

---

## 2. Những bài học kỹ thuật cốt lõi của LAB 09

### 2.1. Vấn đề Subword Label Alignment và `ignore_index = -100`
- Thực nghiệm trên 20 câu ở `tokenizer_alignment.ipynb` cho thấy **tỉ lệ tách subword ở Entity Words ($R = 1.5075$) cao hơn hẳn Non-Entity Words ($R = 1.1687$)** vì tên người, tên công ty mới (`SpaceX`, `BioNTech`, `Zelenskyy`) thường không nằm trọn trong từ điển gốc của BERT.
- **Strategy A (First-subword + `-100`):** Chỉ tính loss trên subword đầu tiên của mỗi từ, các subword sau mang nhãn `-100`. Cần nhớ rõ: token có nhãn `-100` **vẫn đi qua Self-Attention** để truyền thông tin hình thái và ngữ nghĩa cho subword đầu tiên, nhưng không làm phình trọng số loss đối với các từ bị chia nhiều mảnh.

### 2.2. Tại sao tuyệt đối không dùng Token Accuracy để đánh giá NER?
- Trong CoNLL-2003, trên $80\%$ token là nhãn `O`. Một mô hình dự đoán toàn bộ là `O` vẫn đạt Token Accuracy $> 80\%$ dù không tìm ra bất kỳ thực thể nào ($F_1 = 0\%$).
- Baseline Logistic Regression đạt Token Accuracy $94.12\%$ nhưng Entity-level F1 chỉ là $73.00\%$. Chỉ khi dùng **Entity-level Exact Match** (yêu cầu đúng cả loại thực thể lẫn điểm đầu và điểm cuối span), ta mới đo đúng chất lượng trích xuất thực tế.

### 2.3. Chuỗi nối từ Static Embedding (LAB 03) $\to$ Contextual Token Prediction (LAB 09)
- Với **Static Embedding (LAB 03)**, từ *"Washington"* hay *"Apple"* chỉ có một vector cố định duy nhất, không thể phân biệt `PER` vs. `LOC` hay `ORG` vs. `O`.
- Với **Contextual Representation (LAB 08–09)**, Self-Attention điều biến biểu diễn từng token theo toàn bộ ngữ cảnh trái và phải, cho phép gán đúng nhãn thực thể cho cùng một từ vựng trong các ngữ cảnh hoàn toàn khác nhau.

---

## 3. Kiểm chứng 3 Giả thuyết ban đầu (Prediction Verification)

| STT | Giả thuyết ban đầu | Dự đoán | Kết quả thực nghiệm | Đánh giá | Giải thích |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Pretrained Transformer vs. Word-Feature Baseline? | BERT vượt trội mạnh về Entity F1 | BERT $F_1 = 91.23\%$ vs. Baseline $F_1 = 73.00\%$ ($+18.23\%$) | **ĐÚNG** | Ngữ cảnh hai chiều và tri thức tiền huấn luyện giúp giải quyết từ đa nghĩa và ranh giới cụm từ vượt xa đặc trưng cục bộ. |
| **2** | Thực thể đa token dài khó hơn thực thể 1 token? | Khó hơn, F1 giảm dần theo độ dài | BERT F1: $93.67\%$ (1 tok) $\to 90.67\%$ (2) $\to 84.65\%$ (3) $\to 74.99\%$ (4+) | **ĐÚNG** | Khớp chính xác toàn span làm tích lũy xác suất lỗi biên, đặc biệt khi có giới từ/từ nối bên trong tên tổ chức. |
| **3** | Thực thể hiếm gây lỗi nhiều hơn? | Có, tăng FN và sai loại thực thể | Các ca `BioNTech`, `Skonto`, `FTSE` đều bị bỏ sót hoặc nhận sai | **ĐÚNG** | Thực thể đuôi dài thiếu tiên nghiệm từ vựng trong tập train và thường bị WordPiece chia nhỏ thành nhiều subwords `##`. |

---

## 4. Giới hạn còn tồn đọng của NER dạng BIO phẳng & Bước chuyển sang Information Extraction

1. **Thực thể lồng nhau (Nested Entities):** Chuỗi BIO phẳng chỉ cho phép gán đúng 1 nhãn cho mỗi token, nên không thể biểu diễn đồng thời `California` (`LOC`) nằm bên trong `University of California Berkeley` (`ORG`).
2. **Đứt gãy chuyển tiếp nhãn:** Lớp tuyến tính độc lập trên từng token đôi khi sinh chuỗi phi logic (ví dụ `O -> I-ORG` không có `B-ORG`), có thể khắc phục bằng tầng **CRF**.
3. **Vai trò trong Information Extraction (IE):** Như đã kiểm chứng trên 30 câu tin tức (Entity F1 $= 96.55\%$ với Baseline và $97.93\%$ với BERT), NER là bước nền tảng biến văn bản phi cấu trúc thành các đối tượng có cấu trúc (`PERSON`, `ORG`, `LOCATION`), mở đường cho Relation Extraction và Knowledge Graph Construction.

---

## 5. Tuyên bố sử dụng AI (AI Usage Statement)

Căn cứ quy định tại **Mục 28. AI Usage**:

```yaml
AI-assisted code:
  Tool: Antigravity IDE (DeepMind Agentic Assistant)
  What was generated:
    - Khung hàm vẽ đồ thị Matplotlib độ phân giải cao (generate_w9_plots.py).
    - Khung cấu trúc tổ chức bảng dữ liệu thực nghiệm và boilerplate DataLoader/JSON export.
  What was modified:
    - Loại bỏ toàn bộ comment code trong các tệp Python và code cell của Notebook theo quy chuẩn nghiêm ngặt.
    - Cấu hình tải tokenizer và BertConfig hoàn toàn offline (local_files_only=True) và chuẩn hóa hàm extract_entities khớp chuẩn CoNLL.
  What was verified:
    - Sinh viên trực tiếp kiểm tra lời giải BIO tagging, Entity boundary, Tensor shapes tại calculations.md.
    - Kiểm thử thực thi độc lập baseline_ner.py, information_extraction.py (trên 30 câu gold benchmark) và thực chạy tokenizer trên 20 câu để xác nhận tỉ số subword R = 1.5075 (Entity) vs 1.1687 (Non-Entity).
```
