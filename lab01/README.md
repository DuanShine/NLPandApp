## Giới thiệu tổng quan lab01

- **Biểu diễn văn bản truyền thống**: Tiền xử lý (Tokenization, Lemmatization, Stopwords), Thống kê tần suất (Count Vectorizer, Bag-of-Words, TF-IDF), Khảo sát tính thưa (Sparsity).
- **Truy vấn thông tin & Tìm kiếm**: Xây dựng Search Engine dựa trên không gian vector (Vector Space Model), Cosine Similarity, Đánh giá định lượng ($P@K$, $R@K$, $MRR$), Phân tích lỗi (Error Analysis).

- ##  Cấu trúc thư mục Lab 1

Thư mục `lab01/` chứa toàn bộ mã nguồn, dữ liệu thực nghiệm và báo cáo của Bài thực hành 1:

```text
lab1/
├── experiments.ipynb       # Notebook chính: toàn bộ thực nghiệm từ Part D đến Part J 
├── implementation.py       # Module chứa các hàm tính toán cốt lõi (TF, IDF, TF-IDF, Cosine Similarity)
├── prediction.pdf          # Bản scan báo cáo tính toán và dự đoán trước khi thực nghiệm (Part B Calculation and Part C — Predictions)
├── Reflection.md           # Báo cáo tự đánh giá, rút kinh nghiệm và tổng kết bài học (Part 16)
├── results.csv             # Bảng kết quả truy vấn và đo lường định lượng (P@5, R@5, MRR)
├── .gitignore              # Cấu hình bỏ qua file nháp (lab1_v2.ipynb), file cũ và bộ nhớ đệm
└── README.md               # Tài liệu hướng dẫn và báo cáo chi tiết bài thực hành Lab 1
