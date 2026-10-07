# LAB 03 — Word Embeddings & Distributed Representations

## Giới thiệu tổng quan Lab 03
Lab 03 tập trung vào việc nghiên cứu và thực hành các phương pháp Biểu diễn từ phân tán (Distributed Word Representations), từ Ma trận đồng xuất hiện (Co-occurrence Matrix) đến mô hình Word2Vec (CBOW và Skip-gram).

Nội dung chính của Lab 03 gồm ba phần:
- **Mô hình biểu diễn & Huấn luyện**: Triển khai mã nguồn thuần túy cho Co-occurrence Matrix và Word2Vec (CBOW & Skip-gram), giải quyết nút thắt biểu diễn thưa (Sparse bottleneck) để chuyển sang không gian vector đặc số chiều thấp (Dense Embeddings).
- **Thực nghiệm & Ứng dụng**: Khảo sát ảnh hưởng của các siêu tham số (Context Window $w \in \{2, 5, 10\}$, Dimension $d \in \{50, 100, 300\}$), đánh giá độ tương đồng Cosine, thực hiện bài toán Analogy ($king - man + woman \approx queen$) và triển khai ứng dụng Semantic Search.
- **Đánh giá & Hạn chế**: Phân tích lỗi định tính, đánh giá sự sụp đổ không gian ngữ nghĩa khi xử lý từ đa nghĩa (Polysemy như từ "bank"), qua đó làm tiền đề chuyển dịch sang các mô hình ngữ cảnh động (Contextual Embeddings / Transformer).

## Cấu trúc thư mục Lab 03
Thư mục `w3/` chứa toàn bộ mã nguồn, dữ liệu thực nghiệm và tài liệu báo cáo của bài thực hành:

```text
lab03/
├── README.md             # Báo cáo tổng quan, kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.md       # Tính toán chi tiết phép toán Analogy và phân tích dung lượng bộ nhớ
├── prediction.md         # Phân tích mẫu huấn luyện CBOW vs Skip-gram và thiết lập giả thuyết
├── cooccurrence.py       # Module mã nguồn thuần triển khai Co-occurrence Matrix & Similarity
├── word_embedding.ipynb  # Notebook thực nghiệm hoàn chỉnh từ Mục 10 đến Mục 27
├── results.csv           # Dữ liệu định lượng kết quả đo lường tương đồng từ vựng
├── error_analysis.md     # Bảng phân tích chuyên sâu các trường hợp đúng và sai/bất ngờ
└── reflection.md         # Bảng so sánh 4 thế hệ biểu diễn và trả lời 6 câu hỏi vấn đáp
