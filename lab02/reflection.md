#### Câu 1: Nếu tăng độ dài N-gram, mô hình nhận thêm thông tin gì?

Tăng bậc N-gram giúp mô hình sử dụng thêm thông tin từ các từ đứng trước khi dự đoán từ tiếp theo. Nhờ đó, mô hình có thể nhận biết được nhiều mẫu ngữ pháp và cách kết hợp từ hơn.

Ví dụ, Unigram chỉ xét từng từ riêng lẻ, Bigram xét thêm một từ đứng trước và Trigram xét hai từ đứng trước. Vì vậy, N càng lớn thì mô hình càng có thêm thông tin về ngữ cảnh gần.

Tuy nhiên, thông tin mà N-gram sử dụng vẫn bị giới hạn trong một cửa sổ cố định. Mô hình không thể sử dụng những từ nằm ngoài cửa sổ này.

---

#### Câu 2: Tại sao tăng độ dài N-gram lại làm sự thưa thớt dữ liệu tăng lên?

Khi N tăng, số lượng tổ hợp N-gram có thể tạo ra cũng tăng rất nhanh theo kích thước từ vựng.

Nếu kích thước từ vựng là `V`, số lượng tổ hợp N-gram về mặt lý thuyết có thể lên tới:

$$
V^N
$$

Khi N càng lớn, không gian các tổ hợp càng rộng. Trong khi đó, dữ liệu huấn luyện chỉ chứa một phần nhỏ trong số các tổ hợp này.

Do đó, càng tăng N thì càng có nhiều N-gram chưa từng xuất hiện trong tập huấn luyện. Đây chính là nguyên nhân làm dữ liệu trở nên thưa hơn và khiến việc ước lượng xác suất gặp nhiều khó khăn.

---

#### Câu 3: Tại sao kỹ thuật làm mịn lại cần thiết?

Làm mịn được sử dụng để xử lý các N-gram chưa từng xuất hiện trong tập huấn luyện.

Nếu sử dụng MLE mà một N-gram chưa từng xuất hiện, xác suất của nó sẽ bằng `0`. Khi tính xác suất cho cả câu, chỉ cần một N-gram có xác suất bằng `0` thì xác suất của toàn bộ câu cũng trở thành `0`.

Khi tính Perplexity theo dạng logarit:

$$
\log(0) \rightarrow -\infty
$$

Điều này khiến Perplexity của câu hoặc toàn bộ tập dữ liệu trở thành vô cực.

Các phương pháp làm mịn như Laplace Smoothing giúp gán một xác suất nhỏ khác `0` cho những N-gram chưa từng xuất hiện, từ đó tránh được vấn đề này.

---

#### Câu 4: Điểm số Perplexity đo lường điều gì?

Perplexity thể hiện mức độ khó khăn của mô hình khi dự đoán các từ tiếp theo trên một tập dữ liệu.

Có thể hiểu đơn giản rằng Perplexity cho biết mô hình đang phải phân biệt giữa bao nhiêu khả năng ở mỗi bước dự đoán.

Perplexity càng thấp thì mô hình càng đưa ra xác suất cao cho những từ thực sự xuất hiện trong dữ liệu.

Tuy nhiên, Perplexity thấp không phải lúc nào cũng đồng nghĩa với việc mô hình hiểu ngôn ngữ tốt hơn. Cần đánh giá trên dữ liệu phù hợp và kết hợp với các tiêu chí khác.

---

#### Câu 5: Một mô hình có điểm Perplexity thấp hơn có luôn sinh ra văn bản tốt hơn đối với con người không?

Không phải lúc nào cũng vậy.

Perplexity chủ yếu đánh giá khả năng dự đoán từ tiếp theo dựa trên phân phối xác suất của mô hình. Nó không trực tiếp đo lường chất lượng của văn bản về mặt ý nghĩa, tính logic hay mức độ phù hợp với cách con người sử dụng ngôn ngữ.

Một mô hình có Perplexity thấp trên tập huấn luyện nhưng cao trên dữ liệu mới có thể đang gặp vấn đề quá khớp dữ liệu.

Ngoài ra, hai mô hình có Perplexity gần nhau vẫn có thể tạo ra văn bản khác nhau về nội dung, tính mạch lạc và khả năng truyền đạt ý nghĩa.

---

#### Câu 6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?

N-gram chủ yếu dựa trên thống kê tần suất xuất hiện của các chuỗi từ. Mô hình không thực sự biểu diễn ý nghĩa của câu theo cách con người hiểu.

Một hạn chế lớn của N-gram là khả năng sử dụng ngữ cảnh bị giới hạn bởi giá trị N. Khi một thông tin nằm ngoài cửa sổ N-gram, mô hình không thể sử dụng trực tiếp thông tin đó để dự đoán từ tiếp theo.

Ví dụ, với Trigram, mô hình chỉ xét hai từ ngay trước từ cần dự đoán. Vì vậy, những thông tin xuất hiện từ xa hơn trong câu hoặc đoạn văn không được đưa trực tiếp vào quá trình dự đoán.

Điều này khiến N-gram gặp khó khăn khi xử lý các quan hệ ngữ pháp hoặc ngữ nghĩa kéo dài qua nhiều từ.

---

#### Câu 7: Nếu ngữ cảnh dài 100 từ, Trigram có sử dụng được thông tin của 97 từ đầu không?

Không.

Trigram chỉ sử dụng hai từ đứng ngay trước từ cần dự đoán. Theo giả định của mô hình, xác suất của từ tiếp theo chỉ phụ thuộc vào hai từ trước đó:

$$
P(w_i|w_1,\ldots,w_{i-1})
\approx
P(w_i|w_{i-2},w_{i-1})
$$

Do đó, nếu có một ngữ cảnh dài 100 từ thì khi dự đoán từ tiếp theo, Trigram chỉ sử dụng hai từ gần nhất. Các từ còn lại không được sử dụng trực tiếp trong phép tính xác suất.

Đây là một hạn chế quan trọng của N-gram language model. Khi cần xử lý ngữ cảnh dài hơn, các mô hình như RNN và sau đó là Transformer được phát triển để có khả năng sử dụng thông tin từ phạm vi ngữ cảnh rộng hơn.