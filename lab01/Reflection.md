# Reflection - Lab 01: TF-IDF & Vector Space Model

**1. Prediction nào của em sai?**
Ở phần dự đoán ban đầu, em cho rằng thuật toán TF-IDF sẽ giải quyết được triệt để các hạn chế của Count Vectorizer và luôn đưa các tài liệu có ý nghĩa nhất lên đầu. Tuy nhiên, thực tế chứng minh điều này sai. Dù đã có trọng số IDF để phạt các từ phổ biến, hệ thống vẫn thất bại hoàn toàn trước các từ đồng nghĩa và dễ bị thao túng bởi tần suất lặp từ (TF).

**2. Kết quả nào bất ngờ nhất?**
Điều bất ngờ nhất là hiện tượng "Spam từ khóa" (Keyword Stuffing) dễ dàng đánh bại nội dung chất lượng. Cụ thể, khi truy vấn `climate change warming`, tài liệu đứng Top 1 (Doc 19142) lại là một đoạn text ngắn lặp đi lặp lại cụm từ khóa một cách thiếu tự nhiên. Do chiều dài văn bản ngắn và TF cao, Cosine Similarity của nó vọt lên trên cả những bài viết phân tích chuyên sâu về hiệu ứng nhà kính (vốn dùng từ vựng đa dạng hơn nên TF của từng từ cụ thể lại thấp hơn).

**3. Experiment nào cung cấp evidence mạnh nhất?**
Phần **Error Analysis** cung cấp bằng chứng thực nghiệm mạnh mẽ nhất. Thay vì chỉ nhìn vào các con số P@5 hay R@5 trung bình, việc trích xuất và đọc trực tiếp raw text của các tài liệu Top 5 đã phơi bày rõ ràng cơ chế "đếm chữ vô hồn" của thuật toán, cho thấy tại sao một document có nội dung rác lại lọt top, hoặc một document chất lượng lại bị đánh trượt.

**4. Failure case quan trọng nhất là gì?**
Đó là **Khoảng trống ngữ nghĩa (Semantic Gap)** do giới hạn của Lexical Matching. Ở truy vấn `database sql query`, hệ thống đánh giá rất thấp (hoặc bỏ qua) các tài liệu hướng dẫn viết "SQL commands" hay "stored procedure". Thuật toán hoàn toàn mù tịt về mặt ngữ nghĩa; nó chấm điểm 0 cho sự tương đồng chỉ vì từ `query` và từ `commands` không khớp nhau từng chữ cái một, dù trong chuyên ngành chúng biểu đạt cùng một hành động.

**5. Nếu được xây lại search engine, em sẽ thay đổi điều gì?**
Em sẽ chuyển sang mô hình **Hybrid Search** (Tìm kiếm lai):
*   **Tiền xử lý:** Bổ sung Lemmatization (để quy các từ như `algorithms` và `algorithm` về chung một vector).
*   **Lexical:** Nâng cấp TF-IDF lên **BM25** để xử lý bài toán bão hòa tần suất (TF saturation) và độ dài tài liệu, ngăn chặn việc spam từ khóa.
*   **Semantic:** Tích hợp thêm **Dense Retrieval** sử dụng Word Embeddings (như BERT) để ánh xạ các từ đồng nghĩa vào chung một không gian, giúp hệ thống "hiểu" ngữ cảnh thay vì chỉ khớp mặt chữ.

**6. AI đã được sử dụng ở những phần nào và đóng góp cụ thể là gì?**
Em đã sử dụng AI làm đối tác tư duy (thought partner) trong các khâu:
*   **Thiết kế Evaluation Set:** AI cung cấp chiến thuật "Reverse Engineering" (random lấy text đọc lướt rồi mới đặt Query ngược lại), giúp tạo Ground Truth chất lượng mà không phải đọc mù mờ trong 30.000 documents.
*   **Coding Assistant & Debug:** Viết nhanh các đoạn script Python để trích xuất preview text từ danh sách Document ID, tự động hóa khâu tính toán các metric P@5, R@5, MRR và lưu ra file CSV.
*   **Phân tích nguyên nhân sâu xa:** Hỗ trợ mổ xẻ và gọi tên các điểm mù hệ thống (ví dụ: Lexical vs Semantic matching) dựa trên chính log kết quả text thực tế mà em cung cấp.
