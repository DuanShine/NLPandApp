# Giới thiệu tổng quan Lab 02

Lab 02 tập trung vào việc xây dựng và đánh giá **mô hình ngôn ngữ truyền thống dựa trên N-gram**, qua đó tìm hiểu cách mô hình ước lượng xác suất của các từ trong câu và dự đoán từ tiếp theo dựa trên ngữ cảnh trước đó.

Nội dung chính của Lab 02 gồm ba phần:

* **Mô hình ngôn ngữ truyền thống:** Tìm hiểu phương pháp ước lượng xác suất bằng Maximum Likelihood Estimation, mô hình Markov N-gram gồm Unigram, Bigram và Trigram, cùng kỹ thuật smoothing Laplace để xử lý các N-gram chưa xuất hiện trong dữ liệu.
* **Đánh giá và ứng dụng:** Sử dụng Perplexity để đánh giá khả năng dự đoán của mô hình, phân tích vấn đề dữ liệu thưa và zero-frequency, đồng thời triển khai các ứng dụng như dự đoán từ tiếp theo, xếp hạng câu và phân tích lỗi.
* **Giới hạn và hướng phát triển:** Phân tích ảnh hưởng của độ dài ngữ cảnh đến khả năng mô hình hóa ngôn ngữ của N-gram. Những hạn chế này tạo tiền đề cho việc tìm hiểu các mô hình ngôn ngữ nơ-ron như Neural Language Model, Word Embeddings, RNN và Transformer.

## Cấu trúc thư mục Lab 02

Thư mục `lab02/` chứa toàn bộ mã nguồn, dữ liệu thực nghiệm và các tài liệu báo cáo của bài thực hành:

```text
w2/
├── experiment.ipynb      # Notebook chính: huấn luyện mô hình, tính Perplexity và chạy các ứng dụng
├── ngrams_lm.py          # Module triển khai lớp N-gram Language Model
├── caculator.pdf         # Bản scan các bài tính toán tay về xác suất và Perplexity
├── prediction.pdf        # Bản scan dự đoán kết quả trước khi chạy thực nghiệm
├── error_analysis.md     # Báo cáo phân tích các trường hợp dự đoán sai
├── reflection.md         # Báo cáo tự đánh giá, câu hỏi suy ngẫm và tổng kết bài học
├── result.csv            # Kết quả thực nghiệm về Next-word Prediction và Perplexity
└── README.md             # Tài liệu tổng quan và hướng dẫn chi tiết Lab 02
```
