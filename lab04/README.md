# LAB 04 — Text Classification: Representation, Pipeline & Evaluation

## 1. Tổng quan Lab
Buổi lab này là điểm hội tụ của ba thành phần nền tảng đã xây dựng ở các lab trước[cite: 1]:
- **LAB 01:** Biểu diễn văn bản thưa bằng TF-IDF[cite: 1].
- **LAB 02:** Mô hình hóa xác suất chuỗi bằng N-gram Language Model[cite: 1].
- **LAB 03:** Biểu diễn từ ngữ nghĩa dày đặc bằng Dense Word Embeddings[cite: 1].

Trong LAB 04, các biểu diễn trên được đưa vào một bài toán h mang tính ứng dụng cốt lõi: **Phân loại văn bản (Text Classification)**[cite: 1]. Bạn sẽ đi qua một chu trình nghiên cứu và thực nghiệm hoàn chỉnh[cite: 1]:
Tính toán tay (Logits & Metrics) → Thiết lập giả thuyết dự đoán → Thiết kế pipeline chống Data Leakage → Huấn luyện Baseline (TF-IDF + LR) → So sánh đa biểu diễn (TF-IDF vs Average Word2Vec vs Linear SVM) → Thí nghiệm bóc tách trật tự từ (Word Order Ablation với "not good") → Mở rộng n-gram đặc trưng → Đánh giá Confusion Matrix → Khảo sát trọng số mô hình (Interpretability) → Ứng dụng thực tế → Phân tích lỗi định tính chuyên sâu[cite: 1].

### 10 Mục tiêu học tập cốt lõi (Learning Outcomes):
1. **LO1:** Formulate bài toán phân loại văn bản dưới dạng học có giám sát ($f: x \to y$)[cite: 1].
2. **LO2:** Xây dựng quy trình phân chia Train/Validation/Test split đúng chuẩn khoa học để loại trừ hoàn toàn nguy cơ Data Leakage[cite: 1].
3. **LO3:** Chuyển đổi văn bản thô thành các dạng véc-tơ đặc trưng khác nhau (Sparse TF-IDF vs Dense Average Word Embeddings)[cite: 1].
4. **LO4:** Huấn luyện thành công các bộ phân loại tuyến tính cơ bản (Logistic Regression, Linear SVM)[cite: 1].
5. **LO5:** So sánh định lượng và lý giải bản chất hiệu năng giữa biểu diễn từ vựng cục bộ (TF-IDF) và biểu diễn ngữ nghĩa phân tán (Word2Vec)[cite: 1].
6. **LO6:** Tính toán, diễn giải tường minh ý nghĩa hình học và nghiệp vụ của Confusion Matrix, Precision, Recall, F1-score[cite: 1].
7. **LO7:** Nhận diện và hóa giải Bẫy độ chính xác (The Accuracy Trap) trong bối cảnh dữ liệu mất cân bằng (Class Imbalance)[cite: 1].
8. **LO8:** Thực hiện phân tích lỗi (Error Analysis) định tính trên các mẫu dự đoán sai thực tế dựa trên 8 nhóm nguyên nhân kinh điển[cite: 1].
9. **LO9:** Khảo sát tính khả giải thích của mô hình thông qua việc phân tích trọng số các từ khóa tích cực và tiêu cực hàng đầu[cite: 1].
10. **LO10:** Nhận diện giới hạn cố hữu của mô hình túi từ (Bag-of-Words) trước hiện tượng phủ định ("not good"), tạo tiền đề vững chắc cho việc chuyển dịch sang các kiến trúc nơ-ron hiện đại (RNN, Transformer)[cite: 1].

---

## 2. Cấu trúc thư mục nộp bài (Deliverables)
Thư mục `lab04/` bao gồm đầy đủ 09 tệp tin theo đúng quy định[cite: 1]:
```text
lab04/
├── README.md             # Báo cáo tổng quan, kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.pdf      # Lời giải chi tiết 4 bài tập tính toán (Logits, Threshold, Confusion Matrix, Accuracy Trap)
├── prediction.pdf        # Thiết lập 5 giả thuyết và dự đoán định lượng trước khi chạy thực nghiệm
├── classification.ipynb  # Notebook thực nghiệm hoàn chỉnh từ Mục 11 đến Mục 23 (không comment trong code)
├── preprocessing.py      # Module tiền xử lý văn bản, split chống leakage, SVD embeddings & metrics thuần
├── results.csv           # Bảng số liệu thực nghiệm định lượng của 4 cấu hình mô hình
├── confusion_matrix.png  # Biểu đồ trực quan hóa ma trận nhầm lẫn của mô hình tối ưu
├── error_analysis.md     # Bảng phân tích chi tiết các ca đúng/sai dựa trên 8 nhóm nguyên nhân
└── reflection.md         # Bảng so sánh 2 trường phái biểu diễn và lời giải 6 câu hỏi vấn đáp cá nhân
