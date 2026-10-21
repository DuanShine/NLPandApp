# LAB 05 — Sequence Models: RNN, LSTM and GRU

### 1. Tổng quan Lab

Lab này tìm hiểu sự chuyển đổi từ các phương pháp biểu diễn văn bản dựa trên túi từ sang các mô hình mạng nơ-ron có khả năng xử lý chuỗi theo thứ tự. Dựa trên biểu diễn từ bằng Dense Embeddings, chúng ta sử dụng các mạng hồi quy để giải quyết bài toán phân loại cảm xúc trên bộ dữ liệu IMDb.

Nội dung thực nghiệm bao gồm tiền xử lý chuỗi bằng Padding và Truncation, xây dựng mô hình Vanilla RNN, LSTM và GRU, so sánh hiệu năng giữa các mô hình, khảo sát ảnh hưởng của độ dài chuỗi đầu vào và phân tích hiện tượng Overfitting qua Learning Curves. Cuối cùng, chúng ta xây dựng quy trình dự đoán hoàn chỉnh và phân tích các trường hợp mô hình phân loại đúng hoặc sai.

### 2. Mục tiêu học tập

1. **LO1 — Sequence Modeling:** Phân biệt cách biểu diễn văn bản bằng Bag-of-Words, Word2Vec và các mô hình xử lý chuỗi dựa trên trạng thái ẩn.

2. **LO2 — RNN Computation Graph:** Hiểu và tính toán quá trình cập nhật trạng thái ẩn của RNN:

   $$
   \vec{h}_t = \tanh(W_{xh}\vec{x}_t + W_{hh}\vec{h}_{t-1} + \vec{b}_h)
   $$

   Đồng thời, hiểu cách sử dụng trạng thái cuối chuỗi để dự đoán nhãn:

   $$
   \hat{y} = \operatorname{softmax}(W_{hy}\vec{h}_T + \vec{b}_y)
   $$

3. **LO3 — Word Order Sensitivity:** Giải thích khả năng xử lý thứ tự từ của RNN qua các ví dụ như `dog bites man` và `man bites dog`, cũng như ảnh hưởng của cấu trúc phủ định như `not like`.

4. **LO4 — Vanishing and Exploding Gradients:** Giải thích nguyên nhân gây ra hiện tượng triệt tiêu và bùng nổ gradient thông qua tích các Jacobian trong thuật toán Backpropagation Through Time:

   $$
   \prod_{k=t+1}^{T} \frac{\partial \vec{h}_k}{\partial \vec{h}_{k-1}}
   $$
5. **LO5 — LSTM Gating Mechanism:** Tìm hiểu ba cổng Forget Gate, Input Gate và Output Gate, cùng Cell State giúp duy trì và điều chỉnh thông tin qua nhiều bước thời gian.

6. **LO6 — GRU Simplification:** So sánh kiến trúc GRU với LSTM, tập trung vào Update Gate, Reset Gate và cơ chế cập nhật trạng thái ẩn mà không cần Cell State riêng biệt.

7. **LO7 — Sequence Preprocessing:** Phân biệt Padding và Truncation, hiểu vai trò của Padding Mask khi cần thiết, đồng thời đánh giá sự đánh đổi giữa lượng thông tin được giữ lại và chi phí tính toán theo độ dài chuỗi.

8. **LO8 — Empirical Performance Comparison:** So sánh các mô hình TF-IDF, Vanilla RNN, LSTM và GRU dựa trên Accuracy, Precision, Recall, F1-score, thời gian huấn luyện và số lượng tham số.

9. **LO9 — Model Diagnosis:** Phân tích Learning Curves để nhận diện dấu hiệu Overfitting, đặc biệt khi Train Loss tiếp tục giảm nhưng Validation Loss tăng trở lại.

10. **LO10 — Error Analysis and Attention:** Phân tích các trường hợp dự đoán sai liên quan đến phụ thuộc dài hạn, câu châm biếm và thông tin bị mất khi cắt chuỗi. Từ đó, tìm hiểu giới hạn của trạng thái ẩn trong việc lưu giữ thông tin và vai trò của Attention trong các mô hình xử lý chuỗi.

### 3. Cấu trúc thư mục nộp bài

Thư mục `lab05/` gồm 09 tệp phục vụ cho báo cáo, thực nghiệm và đánh giá mô hình.

```text
lab05/
├── README.md               # Báo cáo tổng quan, kết quả thực nghiệm và hướng dẫn sử dụng
├── calculations.pdf        # Lời giải cho bài tập tính tay
├── prediction.pdf          # Thiết lập giả thuyết định lượng trước khi chạy thực nghiệm
├── sequence_models.ipynb   # Notebook thực nghiệm hoàn chỉnh từ Data Loading, Training đến Application
├── dataset_utils.py        # Module Tokenizer, định nghĩa IMDb Dataset, và hàm PadCollate 
├── result.csv              # Bảng số liệu thực nghiệm định lượng toàn diện 
├── confusion_matrix.png    # Biểu đồ trực quan hóa ma trận nhầm lẫn của mô hình tối ưu GRU
├── Learning_curves.png     # Biểu đồ đường theo dõi Train/Validation Loss qua 10 Epochs
└── evaluation_report.md    # Bảng Error Analysis định tính  và lời giải các câu hỏi vấn đáp ngẫu nhiên 
