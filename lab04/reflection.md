# LAB 04 — REFLECTION: SUY NGẪM LÝ THUYẾT VÀ BIỂU DIỄN VĂN BẢN

## 1. Bảng so sánh TF-IDF và Average Word2Vec

| **Đặc tính** | **TF-IDF** | **Average Word2Vec** |
|---|---|---|
| **Cơ sở lý thuyết** | Dựa trên tần suất từ và mức độ phổ biến của từ trong tập tài liệu. | Dựa trên giả thuyết phân bố ngôn ngữ, theo đó các từ xuất hiện trong ngữ cảnh tương tự thường có ý nghĩa gần nhau. |
| **Không gian vector** | Vector thưa, số chiều bằng kích thước từ vựng. | Vector đặc, số chiều cố định, thường từ 100 đến 300 tùy mô hình. |
| **Cách biểu diễn** | Mỗi chiều tương ứng với một từ trong từ vựng. | Mỗi từ được biểu diễn bằng một vector nhúng, sau đó lấy trung bình các vector để biểu diễn văn bản. |
| **Cách đánh trọng số** | Giảm trọng số của các từ xuất hiện phổ biến trong nhiều tài liệu và làm nổi bật các từ có tính phân biệt. | Các từ được cộng vào vector trung bình mà không tự động điều chỉnh theo IDF. |
| **Quan hệ giữa các từ** | Các từ như `good` và `great` nằm ở hai chiều riêng biệt, không thể hiện trực tiếp sự tương đồng về ngữ nghĩa. | Các từ có ngữ cảnh sử dụng tương tự thường có vector gần nhau. |
| **Chi phí tính toán** | Tạo đặc trưng nhanh, nhưng số chiều có thể rất lớn. | Cần có mô hình Word2Vec đã huấn luyện. Việc tạo vector văn bản khá nhanh sau khi có mô hình. |
| **Khả năng tổng quát hóa** | Khó xử lý các từ chưa có trong từ vựng hoặc các từ đồng nghĩa không xuất hiện trong tập huấn luyện. | Có thể tận dụng sự tương đồng giữa các từ đã có vector, nhưng không tự hiểu được mọi từ đồng nghĩa hoặc từ chưa biết. |
| **Hiệu quả trên bài toán cảm xúc** | Kết quả thực nghiệm đạt 82.00, có lợi thế khi các từ khóa cảm xúc mang tính phân biệt cao. | Kết quả thực nghiệm đạt 75.00, có thể làm giảm ảnh hưởng của các từ cảm xúc khi lấy trung bình toàn câu. |

---

## 2. Vì sao hai phương pháp cho kết quả khác nhau?

Sự khác biệt về kết quả có thể xuất phát từ cách mỗi phương pháp biểu diễn từ ngữ và mức độ phù hợp với bài toán phân loại cảm xúc.

### TF-IDF giữ lại thông tin của các từ quan trọng

Trong một bài review phim dài, phần lớn nội dung có thể dùng để mô tả nhân vật, bối cảnh và cốt truyện. Cảm xúc của người viết đôi khi chỉ được thể hiện qua một số từ như `masterpiece`, `boring` hay `unwatchable`.

TF-IDF giảm trọng số của những từ xuất hiện trong nhiều tài liệu và làm nổi bật các từ có khả năng phân biệt giữa các lớp. Nhờ đó, Logistic Regression có thể học trọng số riêng cho từng từ và sử dụng chúng để phân loại cảm xúc.

### Average Word2Vec có thể làm giảm ảnh hưởng của từ khóa

Average Word2Vec tạo vector văn bản bằng cách lấy trung bình các vector từ:

$$
\vec{x}_{doc}=\frac{1}{n}\sum_{i=1}^{n}\vec{w}_i
$$

Trong phép tính này, các từ đều đóng góp vào vector chung. Một từ thể hiện cảm xúc mạnh như `brilliant` có thể bị ảnh hưởng bởi nhiều từ trung tính như `movie`, `actor` hay `screen`.

Khi văn bản dài và chứa nhiều từ ít liên quan đến cảm xúc, vector trung bình có thể không còn thể hiện rõ sắc thái ban đầu. Đây là một nguyên nhân có thể giải thích vì sao Average Word2Vec đạt kết quả thấp hơn TF-IDF trong thực nghiệm.

Tuy nhiên, điều này không có nghĩa Word2Vec luôn kém hơn TF-IDF. Kết quả còn phụ thuộc vào dữ liệu, cách huấn luyện vector từ và bộ phân loại được sử dụng.

---

## 3. Thí nghiệm với câu "this movie is not good"

Trong thực nghiệm ở Mục 17, độ tương đồng cosine giữa hai câu đạt 0.9898:

- `this movie is good`
- `this movie is not good`

Hai câu có ý nghĩa cảm xúc khác nhau nhưng vector Average Word2Vec lại rất gần nhau.

### Nguyên nhân

**Thứ nhất, phép lấy trung bình không giữ được thứ tự từ.**

Average Word2Vec chỉ lấy trung bình các vector từ, nên hai câu có nhiều từ chung thường tạo ra các vector gần nhau. Phương pháp này không biểu diễn trực tiếp mối quan hệ giữa từ phủ định `not` và từ `good`.

**Thứ hai, từ phủ định không tự động đảo ngược ý nghĩa của từ đứng sau.**

Trong câu `not good`, vector của `not` chỉ đóng góp thêm vào phép tính trung bình. Nó không hoạt động như một phép toán làm đảo ngược ý nghĩa của `good` thành `bad`.

Vì vậy, vector của hai câu vẫn có thể rất giống nhau dù ý nghĩa khác biệt.

**Thứ ba, các mô hình xử lý ngôn ngữ có khả năng biểu diễn ngữ cảnh tốt hơn.**

- **RNN và LSTM:** Xử lý từ theo trình tự và duy trì trạng thái qua các bước thời gian.
- **Transformer:** Sử dụng cơ chế Self-Attention để mô hình hóa mối quan hệ giữa các từ trong câu.
- **BERT:** Tạo biểu diễn từ phụ thuộc vào ngữ cảnh, giúp phân biệt cách sử dụng từ trong những câu khác nhau.

Các mô hình này có khả năng xử lý phủ định tốt hơn nếu được huấn luyện phù hợp, nhưng không phải lúc nào cũng phân loại đúng mọi trường hợp.

---

## 4. Sáu câu hỏi ôn tập lý thuyết

### Câu 1: Tại sao Accuracy 95% chưa chắc đã là một mô hình tốt?

Accuracy đo tỷ lệ dự đoán đúng trên tổng số mẫu:

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

Khi dữ liệu mất cân bằng, Accuracy có thể gây hiểu nhầm. Ví dụ, nếu 95% mẫu thuộc lớp Negative và 5% thuộc lớp Positive, mô hình luôn dự đoán Negative vẫn đạt Accuracy 95%.

Tuy nhiên, mô hình đó không nhận diện được bất kỳ mẫu Positive nào. Recall của lớp Positive bằng 0.

Do đó, ngoài Accuracy, cần xem xét Confusion Matrix, Precision, Recall, F1 Score và Balanced Accuracy để đánh giá mô hình toàn diện hơn.

### Câu 2: TF-IDF khác Average Word2Vec ở những điểm nào?

TF-IDF biểu diễn văn bản bằng vector thưa, có số chiều bằng kích thước từ vựng. Mỗi chiều tương ứng với một từ và giá trị thể hiện mức độ quan trọng của từ đó trong văn bản.

Ngược lại, Average Word2Vec tạo vector đặc có số chiều cố định bằng cách lấy trung bình các vector nhúng của những từ trong câu. Các vector từ được học từ ngữ cảnh sử dụng, nhờ đó có thể thể hiện sự tương đồng về ngữ nghĩa.

TF-IDF có lợi thế khi các từ khóa cụ thể mang tính phân biệt cao. Average Word2Vec tạo ra biểu diễn gọn hơn nhưng có thể làm mất thông tin về thứ tự từ và giảm ảnh hưởng của những từ cảm xúc quan trọng.

### Câu 3: Tại sao cụm từ "not good" có thể gây lỗi cho các mô hình truyền thống?

Với Unigram TF-IDF, từ `not` và từ `good` được xem là hai đặc trưng riêng biệt. Nếu trọng số tích cực của `good` lớn hơn ảnh hưởng tiêu cực của `not`, mô hình có thể dự đoán sai câu thành Positive.

Với Average Word2Vec, vector của hai từ được cộng vào phép tính trung bình. Phương pháp này không trực tiếp biểu diễn quan hệ phủ định giữa chúng, nên vector của câu vẫn có thể gần với câu `this movie is good`.

Bigram có thể cải thiện vấn đề bằng cách bổ sung đặc trưng `not good`, cho phép mô hình học trọng số riêng cho cụm từ này. Tuy nhiên, hiệu quả còn phụ thuộc vào dữ liệu huấn luyện và cách tiền xử lý.

### Câu 4: Data Leakage xảy ra như thế nào nếu fit TF-IDF trước khi chia Train và Test?

Nếu fit TF-IDF trên toàn bộ dữ liệu trước khi chia tập, thông tin từ tập Test sẽ được sử dụng trong quá trình xây dựng bộ trích xuất đặc trưng.

Có hai vấn đề chính:

1. **Rò rỉ từ vựng:** Từ điển có thể chứa những từ chỉ xuất hiện trong tập Test.
2. **Rò rỉ thông tin thống kê:** IDF được tính từ toàn bộ tài liệu, bao gồm cả tập Test, khiến trọng số đặc trưng phụ thuộc vào dữ liệu đánh giá.

Điều này không nhất thiết làm điểm số tăng trong mọi trường hợp, nhưng khiến quy trình đánh giá không còn độc lập hoàn toàn.

Cách làm đúng là chia dữ liệu trước, sau đó chỉ fit TF-IDF trên tập Train và sử dụng `transform` cho tập Validation và Test.

### Câu 5: Precision và Recall khác nhau như thế nào?

**Precision** cho biết tỷ lệ dự đoán Positive chính xác trong tất cả các mẫu được dự đoán là Positive.

$$
Precision=\frac{TP}{TP+FP}
$$

Chỉ số này cần được ưu tiên khi muốn hạn chế False Positive, chẳng hạn khi hệ thống lọc email và không muốn đánh dấu nhầm email quan trọng thành thư rác.

**Recall** cho biết tỷ lệ mẫu Positive thực tế được mô hình nhận diện đúng.

$$
Recall=\frac{TP}{TP+FN}
$$

Chỉ số này cần được ưu tiên khi bỏ sót mẫu Positive gây hậu quả nghiêm trọng, chẳng hạn trong sàng lọc bệnh hoặc phát hiện gian lận.

Trong thực tế, việc lựa chọn Precision hay Recall phụ thuộc vào chi phí của từng loại lỗi. F1 Score có thể được sử dụng khi cần cân bằng hai chỉ số này.

### Câu 6: Tại sao thêm Bigram có thể cải thiện phân loại cảm xúc?

Bigram bổ sung các cặp từ liên tiếp vào tập đặc trưng. Nhờ đó, mô hình có thể phân biệt những cụm từ mà ý nghĩa khác với từng từ đứng riêng lẻ.

Ví dụ:

- `not good`
- `very bad`
- `waste of time`
- `must see`

Với `ngram_range=(1, 2)`, TF-IDF sử dụng cả Unigram và Bigram. Logistic Regression có thể học trọng số riêng cho từng cụm từ, giúp nhận diện một số cấu trúc phủ định hoặc nhấn mạnh tốt hơn.

Tuy nhiên, Bigram làm tăng số lượng đặc trưng và chi phí tính toán. Nó cũng không giải quyết triệt để mọi vấn đề về ngữ cảnh, đặc biệt khi các từ liên quan không nằm cạnh nhau.

---

## 5. Tuyên bố sử dụng công cụ AI

- **Công cụ sử dụng:** Antigravity IDE với Gemini 3.8 Flash.
- **Mục đích hỗ trợ:** Hỗ trợ chỉnh sửa cấu trúc code trong `preprocessing.py`, tạo script chạy thực nghiệm, vẽ biểu đồ Confusion Matrix và định dạng tài liệu Markdown.
- **Nội dung có sử dụng hỗ trợ AI:** Khung notebook `classification.ipynb` và một số phần xử lý ma trận thưa bằng TruncatedSVD trong `preprocessing.py`.
- **Nội dung được sinh viên chỉnh sửa:** Điều chỉnh tham số `V=5000`, thiết lập cách chia Train/Validation/Test và xây dựng bảng so sánh kết quả.
- **Nội dung sinh viên tự thực hiện và kiểm chứng:** Các phép tính trong `calculations.md`, giả thuyết trong `prediction.md`, phân tích lỗi trong `error_analysis.md` và phần trả lời lý thuyết trong `reflection.md`.

Nội dung trên cần được điều chỉnh cho đúng với quá trình thực hiện thực tế và quy định sử dụng AI của môn học.
