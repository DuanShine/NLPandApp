# LAB 05 — REFLECTION: SEQUENCE MODELING, BPTT & BƯỚC CHUYỂN SANG ATTENTION



## 1. Bảng so sánh tiến hóa 5 kiến trúc biểu diễn và mô hình hóa (LAB 01 $\to$ LAB 05)

Bảng đối chiếu toàn diện 5 mô hình kinh điển dựa trên 7 tiêu chí kiến trúc:

| Thuộc tính (Property) | TF-IDF + LR (LAB 01, 04) | Avg Word2Vec + LR (LAB 03, 04) | Simple RNN (LAB 05) | LSTM (LAB 05) | GRU (LAB 05) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Sparse representation** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Dense representation** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Word order (Trật tự từ)** | Limited (với Bigram) | ❌ | ✅ | ✅ | ✅ |
| **Context-dependent state** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Long dependency handling** | ❌ | ❌ | Limited ($T \le 15$) | Improved ($T \le 150$) | Improved ($T \le 150$) |
| **Computational cost** | Low (Rất nhẹ) | Low (Rất nhẹ) | Higher (Tuần tự $\mathcal{O}(T)$) | Higher (Nhiều gates) | Higher (Ít hơn LSTM) |
| **Representation mechanism** | 1 vector rời rạc / doc | 1 vector trung bình / doc | Hidden state chuỗi | Hidden + Cell state | Hidden state chuỗi |

---

## 2. Trả lời câu hỏi trọng tâm: Đột phá và Giới hạn của Sequence Models

### 2.1. RNN/LSTM/GRU giải quyết được vấn đề gì mà Average Word2Vec không giải quyết tốt?

1. **Bảo toàn hoàn toàn trật tự từ vựng (Word Order Awareness):**
   - Average Word2Vec bị ràng buộc bởi tính chất giao hoán của phép cộng véc-tơ, coi *"this movie is good"* và *"this movie is not good"* có khoảng cách gần như bằng 0 ($\text{Cosine} = 0.9898$).
   - Sequence models xử lý văn bản theo từng bước thời gian $t$. Thứ tự xuất hiện của từ ngữ quyết định đường đi của trạng thái ẩn $h_t$, giúp mô hình phân biệt rạch ròi cấu trúc ngữ pháp, vai trò chủ thể - khách thể (*"dog bites man"* vs *"man bites dog"*), và phạm vi ảnh hưởng của từ phủ định (*"not good"*).
2. **Trạng thái phụ thuộc ngữ cảnh động (Context-dependent States):**
   - Trong Average Word2Vec, vector của từ `like` luôn cố định dù nằm trong câu khẳng định hay câu phủ định.
   - Trong RNN/LSTM/GRU, trạng thái $h_t$ sau khi đọc `like` là hàm số của cả từ hiện tại lẫn toàn bộ lịch sử phía trước: $h_t = f(x_t, h_{t-1})$. Tín hiệu phủ định của từ `not` ở bước trước trực tiếp biến đổi trạng thái của `like` thành một biểu diễn tiêu cực.

### 2.2. RNN/LSTM/GRU vẫn còn những điểm nghẽn nghiêm trọng nào?

Mặc dù khắc phục được điểm yếu mất trật tự từ của Bag-of-Words, các mạng nơ-ron hồi quy vẫn vấp phải 3 rào cản cốt tử:

1. **Điểm nghẽn thông tin nén (Information Bottleneck):**
   - Toàn bộ nội dung của một bài văn dài hàng trăm, hàng nghìn từ buộc phải nén tuần tự vào một vector trạng thái ẩn cuối cùng $h_T$ có kích thước cố định (ví dụ $d = 128$).
   - Các chi tiết quan trọng ở đầu câu và giữa câu rất dễ bị "ghi đè" (overwritten) hoặc phai nhạt dần khi đến bước cuối cùng.
2. **Tính toán tuần tự không thể song song hóa (Sequential Computation Bottleneck):**
   - Để tính được $h_t$, mô hình bắt buộc phải chờ tính xong $h_{t-1}$.
   - Ràng buộc tuần tự thời gian $\mathcal{O}(T)$ khiến RNN/LSTM/GRU không thể tận dụng sức mạnh tính toán song song hàng nghìn nhân của phần cứng hiện đại (GPU/TPU), khiến thời gian huấn luyện trên dữ liệu lớn trở nên vô cùng chậm chạp.
3. **Giới hạn khoảng cách phụ thuộc (Effective Context Window):**
   - Ngay cả với LSTM và GRU, khả năng duy trì phụ thuộc xa hiếm khi vượt quá $100 - 200$ bước thời gian do gradient vẫn bị suy giảm nhẹ và dung lượng ô nhớ có hạn.
   - Khi gặp các tài liệu pháp lý, sách báo, bài luận dài $1000 - 5000$ từ, LSTM bắt đầu bộc lộ sự bất lực.

$\implies$ **Động lực tiến hóa sang Attention Mechanism & Transformer:**
Cơ chế **Attention (Bahdanau et al., 2015)** và kiến trúc **Transformer (Vaswani et al., 2017)** ra đời để giải phóng hoàn toàn các điểm nghẽn này: cho phép từ hiện tại kết nối trực tiếp với *mọi từ khác* trong câu mà không qua trung gian tuần tự, đồng thời tính toán song song hóa $100\%$ trên toàn bộ chuỗi văn bản.

---

## 3. Trả lời chi tiết 8 câu hỏi kiểm tra vấn đáp cá nhân (Individual Learning Check)

### Câu 1: RNN khác Average Word2Vec ở những điểm cốt lõi nào?
* **Trả lời:**
  - **Cơ chế xử lý:** Average Word2Vec là mô hình tĩnh (Bag-of-Words), cộng dồn và chia trung bình độc lập các véc-tơ từ không quan tâm vị trí. RNN là mô hình chuỗi động, đọc tuần tự từng từ một qua thời gian.
  - **Trật tự từ:** Average Word2Vec hoàn toàn không phân biệt được thứ tự từ. RNN nhạy cảm tuyệt đối với trật tự từ nhờ trạng thái ẩn đệ quy $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$.
  - **Biểu diễn từ:** Word2Vec gán mỗi từ một vector tĩnh duy nhất. RNN tạo ra các vector trạng thái ẩn $h_t$ biến đổi liên tục theo ngữ cảnh xung quanh.

### Câu 2: Hidden state ($h_t$) trong RNN mang ý nghĩa vật lý và toán học gì?
* **Trả lời:**
  Về mặt toán học, $h_t \in \mathbb{R}^d$ là vector đầu ra của hàm phi tuyến tính tổ hợp giữa đầu vào hiện tại $x_t$ và trạng thái quá khứ $h_{t-1}$. Về mặt ngôn ngữ học, $h_t$ đóng vai trò như **bộ nhớ ngắn hạn (Working Memory)** của mô hình: nó nén và lưu trữ toàn bộ lịch sử thông tin ngữ nghĩa, cấu trúc cú pháp của tất cả các từ đã đọc từ bước $1$ đến bước $t$, làm bối cảnh để tiếp nhận từ tiếp theo $x_{t+1}$ hoặc làm cơ sở để phân loại câu tại bước cuối $T$.

### Câu 3: Tại sao Simple RNN lại gặp hiện tượng triệt tiêu gradient (Vanishing Gradient)?
* **Trả lời:**
  Khi tính đạo hàm lan truyền ngược theo thời gian (BPTT), gradient từ bước $T$ truyền về bước $t$ chứa tích chuỗi của $T - t$ ma trận Jacobian: $\prod_{k=t+1}^T \text{diag}(1 - h_k^2) W_{hh}^T$. Vì đạo hàm của hàm $\tanh$ luôn nằm trong khoảng $(0, 1]$ và thường xuyên rơi vào vùng bão hòa gần $0$, kết hợp với các giá trị riêng của ma trận trọng số $W_{hh}$, tích chuỗi này suy giảm về $0$ theo cấp số nhân $(\gamma \cdot \lambda_{\max})^{T-t}$. Khi khoảng cách xa ($T - t > 15$), gradient tiệm cận 0, khiến các trọng số không nhận được tín hiệu lỗi để cập nhật cho các từ ở đầu câu.

### Câu 4: Cổng quên (Forget Gate) của LSTM dùng để làm gì và có ý nghĩa toán học ra sao?
* **Trả lời:**
  Forget gate $f_t = \sigma(W_f [x_t, h_{t-1}] + b_f) \in (0, 1)$ đóng vai trò như một **van điều tiết tỷ lệ giữ lại ký ức cũ** trong ô nhớ: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. 
  - Nếu $f_t \to 1$: Mô hình giữ lại nguyên vẹn thông tin từ bước trước, tạo thành "đường cao tốc gradient" (đạo hàm $\frac{\partial c_t}{\partial c_{t-1}} \approx 1$) giúp truyền tín hiệu học tập qua hàng trăm bước thời gian mà không bị triệt tiêu.
  - Nếu $f_t \to 0$: Mô hình chủ động xóa sạch ký ức cũ không còn liên quan (ví dụ khi kết thúc câu hoặc đổi chủ đề) để ô nhớ sẵn sàng nạp thông tin mới.

### Câu 5: Tại sao tăng Sequence Length có thể làm quá trình huấn luyện chậm hơn?
* **Trả lời:**
  Độ phức tạp tính toán của mạng hồi quy trên một mẫu là tuyến tính theo chiều dài chuỗi: $\mathcal{O}(T \cdot (d \cdot h + h^2))$. Khi tăng $T$ từ $50$ lên $200$, số lượng bước lan truyền thuận (Forward pass) tăng gấp 4 lần, số lượng bước lan truyền ngược (BPTT) cũng tăng gấp 4 lần. Đồng thời, bộ nhớ RAM/VRAM phải lưu trữ toàn bộ đồ thị tính toán của $200$ trạng thái ẩn trung gian, làm tăng chi phí quản lý bộ nhớ và giảm kích thước batch size khả dụng.

### Câu 6: GRU khác LSTM ở những điểm cấu trúc nào?
* **Trả lời:**
  - **Số lượng cổng:** LSTM có 3 cổng riêng biệt (Forget gate, Input gate, Output gate). GRU gộp lại chỉ còn **2 cổng** (Reset gate $r_t$ và Update gate $z_t$).
  - **Trạng thái bộ nhớ:** LSTM duy trì hai luồng trạng thái tách biệt là Hidden state $h_t$ và Cell state $c_t$. GRU loại bỏ Cell state, **chỉ sử dụng duy nhất Hidden state $h_t$**.
  - **Cơ chế cập nhật:** GRU dùng chung cổng update $z_t$ để cân bằng trực tiếp giữa ký ức cũ và thông tin mới: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$.
  - **Số tham số:** GRU có ít hơn khoảng $25\%$ số lượng tham số so với LSTM, giúp huấn luyện nhanh hơn và ít bị overfit hơn trên dữ liệu nhỏ.

### Câu 7: Tại sao mô hình có hiện tượng Training Loss tiếp tục giảm nhưng Validation Loss bắt đầu tăng?
* **Trả lời:**
  Đây là dấu hiệu kinh điển của hiện tượng **Học vẹt / Quá khớp (Overfitting)**. Sau một số epoch nhất định, mô hình đã học xong các quy luật ngôn ngữ mang tính tổng quát chung. Nếu tiếp tục huấn luyện, mô hình bắt đầu "học thuộc lòng" các đặc thù ngẫu nhiên, tiếng nhiễu, lỗi chính tả cục bộ chỉ có trong tập Train để ép Training Loss giảm sâu hơn nữa. Tuy nhiên, các đặc thù này không tồn tại trong tập Validation, khiến cho sai số trên dữ liệu kiểm định Validation Loss bắt đầu quay đầu tăng vọt. Giải pháp là áp dụng **Early Stopping, Dropout, hoặc L2 Regularization**.

### Câu 8: Nếu một bài đánh giá (Review) bị cắt cụt (Truncate) ở 30 tokens đầu tiên, loại thông tin then chốt nào có thể bị mất?
* **Trả lời:**
  Trong văn phong phê bình phim ảnh hoặc sản phẩm, $30$ từ đầu tiên thường chỉ là lời dẫn dắt, giới thiệu tên diễn viên, đạo diễn hoặc tóm tắt sơ lược bối cảnh cốt truyện (thường mang tính trung tính). **Phần kết luận đánh giá mang tính quyết định cảm xúc** (*"Tuy nhiên, kịch bản quá tệ và diễn xuất vô cùng thất vọng"* hoặc *"Tóm lại, đây là một kiệt tác điện ảnh không thể bỏ lỡ"*) hầu như luôn nằm ở phần thân sau hoặc đoạn kết của bài viết. Cắt ở $30$ tokens sẽ làm mất hoàn toàn kết luận cảm xúc thật sự, khiến mô hình chỉ dựa vào các từ dẫn nhập bối cảnh để đoán mò.

---

## 4. Tuyên bố sử dụng Trí tuệ Nhân tạo (AI Assistance Statement)

Theo quy định tại **Mục 27. AI Policy** của đề cương Lab:
- **Công cụ sử dụng (Tool):** Antigravity IDE (Gemini 3.8 Flash Engine).
- **Mục đích hỗ trợ (Purpose):** Hỗ trợ viết code chuẩn hóa cấu trúc `SimpleRNNCell` trong `rnn_cell.py`, sinh biểu đồ learning curves và confusion matrix, tối ưu hóa quá trình tính toán và kiểm tra định dạng toán học LaTeX.
- **Nội dung sinh tự động (Generated):** Khung sườn notebook `sequence_classification.ipynb`, hàm vẽ biểu đồ loss `learning_curves.png`.
- **Nội dung do sinh viên chỉnh sửa & tối ưu (Modified):** Tinh chỉnh cấu hình siêu tham số (Embedding dim $d=64$, Hidden size $h=64$, Epochs $= 5$, Truncation length $50, 100, 200$), thiết lập pipeline tiền xử lý tokenization.
- **Nội dung sinh viên tự thực hiện và kiểm chứng 100% (Verified & Authored):** Toàn bộ lời giải chi tiết 5 bài toán tính toán thuận và cổng LSTM (`calculations.md`), thiết lập 5 giả thuyết nghiên cứu (`prediction.md`), phân tích định tính 6 ca lỗi theo nguyên nhân cấu trúc chuỗi (`error_analysis.md`) và trả lời toàn bộ 8 câu hỏi suy ngẫm lý thuyết chuyên sâu (`reflection.md`).
