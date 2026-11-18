# LAB 09 — Phân tích Lỗi Định tính (Error Analysis): Named Entity Recognition (CoNLL-2003)


---

## 1. Bảng 5 Trường hợp False Positives (FP — Dự đoán thừa hoặc sai loại thực thể)

| Mã | Text (Trích đoạn câu) | Gold | Prediction | Error Type | Explanation |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **FP-1** | *"**SOCCER** - JAPAN GET LUCKY WIN."* | `SOCCER: O` | `SOCCER: B-MISC` | Spurious entity | Từ viết hoa toàn bộ (ALL CAPS) ở tiêu đề tin tức đánh lừa mô hình nhận diện nhầm thành thực thể `MISC`. |
| **FP-2** | *"**England** beat **France** 2-0 in rugby."* | `England: B-ORG`, `France: B-ORG` | `England: B-LOC`, `France: B-LOC` | Wrong entity type | Trong CoNLL-2003, tên quốc gia đại diện cho đội tuyển thể thao mang nhãn `ORG`, nhưng tiên nghiệm từ vựng kéo mô hình về `LOC`. |
| **FP-3** | *"The **Senate** approved the new budget."* | `Senate: B-ORG` | `Senate: B-MISC` | Wrong entity type | Danh từ chỉ cơ quan lập pháp không có tiền tố quốc gia đi kèm dễ gây nhầm lẫn giữa `ORG` và `MISC`. |
| **FP-4** | *"He bought a **Mercury** sedan yesterday."* | `Mercury: B-MISC` | `Mercury: B-ORG` | Ambiguous entity | Mô hình nhầm lẫn giữa tên dòng sản phẩm (`MISC`) và tên hãng sản xuất (`ORG`). |
| **FP-5** | *"**March** sales rose by five percent."* | `March: O` | `March: B-PER` | Spurious entity | Từ chỉ tháng đứng đầu câu viết hoa có hình thức trùng với họ người (*March*), thiếu tín hiệu ngữ cảnh phân định. |

---

## 2. Bảng 5 Trường hợp False Negatives (FN — Bỏ sót thực thể thực tế)

| Mã | Text (Trích đoạn câu) | Gold | Prediction | Error Type | Explanation |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **FN-1** | *"Researchers cited **BioNTech** clinical data."* | `BioNTech: B-ORG` | `BioNTech: O` | Rare entity / Subword alignment | Tên công ty hiếm bị WordPiece chia vụn thành 4 subwords (`bio`, `##nt`, `##ech`), làm loãng biểu diễn ngữ nghĩa của từ. |
| **FN-2** | *"The **2004 Olympics** inspired millions."* | `2004 Olympics: B-MISC I-MISC` | `2004: O`, `Olympics: O` | Missing entity | Cụm sự kiện bắt đầu bằng chữ số (`2004`) ít gặp trong phân phối nhãn `B-MISC` nên mô hình bỏ qua cả cụm. |
| **FN-3** | *"Local club **Skonto** won the league."* | `Skonto: B-ORG` | `Skonto: O` | Rare entity | Thực thể đuôi dài (long-tail entity) không xuất hiện trong tập train và không đi kèm hậu tố tổ chức (`FC`, `Inc.`). |
| **FN-4** | *"She prefers **french** cuisine at home."* | `french: B-MISC` | `french: O` | Context-dependent / Casing | Từ chỉ quốc tịch bị viết thường do lỗi chính tả trong dữ liệu thô làm mất đặc trưng viết hoa quan trọng (với cased/baseline features). |
| **FN-5** | *"Analysts watched **FTSE** closely."* | `FTSE: B-MISC` | `FTSE: O` | Rare entity / Domain shift | Ký hiệu chỉ số tài chính viết tắt chuyên ngành bị chia subwords và thiếu ngữ cảnh giải thích đi kèm. |

---

## 3. Bảng 5 Trường hợp Boundary Errors (BE — Nhận diện sai ranh giới thực thể)

| Mã | Text (Trích đoạn câu) | Gold | Prediction | Error Type | Explanation |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **BE-1** | *"**University of California Berkeley** published a study."* | `University of California Berkeley` (`ORG`) | `University of California` (`ORG`) + `Berkeley` (`LOC`) | Wrong boundary | Thực thể dài 4 tokens chứa tên thành phố ở cuối khiến mô hình cắt đứt ranh giới và tách `Berkeley` thành `B-LOC`. |
| **BE-2** | *"**United Nations Security Council** met today."* | `United Nations Security Council` (`ORG`) | `United Nations` (`ORG`) + `Security Council` (`ORG`) | Wrong boundary | Mô hình chèn thêm thẻ `B-ORG` tại `Security`, tách 1 tổ chức 4 tokens thành 2 tổ chức con (tạo ra 2 FP + 1 FN ở entity-level). |
| **BE-3** | *"He visited the **Bank of England** headquarters."* | `Bank of England` (`ORG`) | `Bank` (`O`), `of` (`O`), `England` (`B-LOC`) | Wrong boundary | Giới từ `"of"` ở giữa gây đứt gãy chuỗi `I-ORG`, mô hình chỉ giữ lại tên quốc gia `England`. |
| **BE-4** | *"**New York Times** reported the deal."* | `New York Times` (`ORG`) | `New York` (`LOC`) + `Times` (`ORG`) | Wrong boundary | Cụm địa danh quen thuộc `New York` kích hoạt mạnh nhãn `LOC`, làm gãy ranh giới của tên tờ báo `ORG`. |
| **BE-5** | *"Spokesperson for **U.S. Department of Defense**."* | `U.S. Department of Defense` (`ORG`) | `U.S.` (`LOC`) + `Department of Defense` (`ORG`) | Wrong boundary / Subword | Từ viết tắt chứa dấu chấm `U.S.` bị tách thành `u`, `.`, `s`, `.`, khiến ranh giới mở đầu `B-ORG` bị dịch sai sang `Department`. |

---

## 4. Tổng hợp nguyên nhân cốt lõi & Hướng khắc phục

1. **Lỗi ranh giới chiếm tỉ trọng lớn nhất trong khoảng cách giữa Token Accuracy và Entity F1:**
   - Trong cả 5 ca Boundary Errors (BE-1 đến BE-5), mô hình nhận diện đúng phần lớn các token nhưng chỉ sai một điểm ngắt `B-`/`I-` hoặc dính bẫy địa danh lồng trong tên tổ chức (*"New York Times"*, *"University of California Berkeley"*).
   - **Khắc phục:** Bổ sung tầng **CRF (Conditional Random Field)** phía trên lớp tuyến tính của BERT để ràng buộc ma trận chuyển tiếp hợp lệ giữa các thẻ `B-X` và `I-X`.
2. **Nhầm lẫn ngữ cảnh thể thao / chính trị (`ORG` vs. `LOC`):**
   - Trong CoNLL-2003, cùng một từ như `England`, `Germany`, `Brazil` vừa là quốc gia (`LOC`) vừa là đội tuyển quốc gia (`ORG`). Cần tăng cường ngữ cảnh cấp đoạn văn (document-level context) thay vì chỉ xử lý từng câu rời rạc.
3. **Phân mảnh Subword trên thực thể hiếm:**
   - Các thực thể mới (*"BioNTech"*, *"Skonto"*) bị tách thành nhiều mảnh `##`, làm giảm khả năng khái quát hóa. Sử dụng mô hình tiền huấn luyện cập nhật (RoBERTa / DeBERTa với Byte-level BPE) hoặc giữ nguyên chữ hoa (`bert-base-cased`) giúp cải thiện rõ rệt.
