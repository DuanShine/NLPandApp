# LAB 06 — Báo cáo Tổng kết & Suy ngẫm (Reflection): Từ Attention đến Transformer


---

## 1. Bảng so sánh toàn diện giữa RNN/LSTM/GRU và Attention Mechanism

| Tiêu chí so sánh | RNN / LSTM / GRU (LAB 05) | Attention Mechanism / Self-Attention (LAB 06) |
| :--- | :--- | :--- |
| **Bản chất tính toán** | Tuần tự từng bước thời gian: $\vec{h}_t = f(\vec{x}_t, \vec{h}_{t-1})$ | Song song hóa hoàn toàn trên toàn chuỗi qua phép nhân ma trận |
| **Biểu diễn chuỗi (Context Representation)** | Trạng thái ẩn nén cố định tại bước cuối $\vec{h}_T$ (Information Bottleneck) | Tổ hợp tuyến tính động có trọng số của các vector giá trị: $\vec{c} = \sum_{i=1}^T \alpha_i \vec{v}_i$ |
| **Trật tự từ (Sequence Order)** | Tự nhiên ngầm định theo dòng thời gian tính toán | **Không có ngầm định**; cần bổ sung Positional Encoding |
| **Khoảng cách tương tác từ - từ (Path Length)** | $\mathcal{O}(T)$ (Phải đi qua $T$ bước chuyển tiếp trạng thái ẩn) | **$\mathcal{O}(1)$** (Bất kỳ hai token nào cũng tương tác trực tiếp qua $q_i^T k_j$) |
| **Xử lý phụ thuộc tầm xa (Long-Range Dependencies)** | Rất khó khăn với RNN (Vanishing Gradient); được cải thiện với LSTM/GRU nhưng vẫn suy giảm | **Rất dễ dàng**; gradient truyền trực tiếp không bị suy giảm theo chiều dài chuỗi |
| **Khả năng song song hóa (Parallel Computation)** | Rất hạn chế trong huấn luyện (Bước $t$ phải chờ bước $t-1$) | **Tối đa trên GPU**; toàn bộ ma trận $Q K^T$ được nhân đồng thời |
| **Độ phức tạp tính toán theo độ dài chuỗi ($T$)** | Tuyến tính: $\mathcal{O}(T \cdot d^2)$ | **Bậc hai: $\mathcal{O}(T^2 \cdot d)$** cho Self-Attention đầy đủ |
| **Dung lượng bộ nhớ lưu trữ ma trận kích hoạt** | $\mathcal{O}(T \cdot d)$ | **$\mathcal{O}(T^2)$** (Lưu ma trận trọng số $T \times T$ cho mỗi đầu Attention) |

---

## 2. Phân tích cốt lõi: Attention giải quyết vấn đề gì và tạo ra thách thức mới nào?

### 2.1. Attention giải quyết vấn đề gì của Recurrent Processing?
1. **Phá vỡ nút cổ chai thông tin (Eliminating the Information Bottleneck):**
   - Trong RNN, toàn bộ nội dung của câu văn dù dài hàng trăm từ đều bị ép buộc phải nén vào một vector $\vec{h}_T$ có kích thước cố định.
   - Attention giải phóng mô hình khỏi nút thắt này bằng cách duy trì toàn bộ các vector giá trị $\{v_1, v_2, \dots, v_T\}$ và cho phép bộ phân loại truy cập linh hoạt vào từng phần thông tin khi cần thiết.
2. **Loại bỏ triệt để hiện tượng Tiêu biến Gradient (Vanishing Gradient across Time):**
   - Đường truyền tín hiệu giữa token thứ $1$ và token thứ $T$ trong RNN đòi hỏi nhân qua chuỗi $T-1$ ma trận Jacobian $\prod_{k=2}^T \frac{\partial \vec{h}_k}{\partial \vec{h}_{k-1}}$.
   - Trong Self-Attention, khoảng cách đường truyền giữa hai token bất kỳ luôn là $\mathcal{O}(1)$. Gradient từ hàm mất mát chảy ngược trực tiếp từ token đích về token nguồn qua phép nhân $Q K^T$ mà không bị suy giảm theo độ dài câu.
3. **Mở khóa năng lực tính toán song song trên phần cứng hiện đại:**
   - RNN không thể tận dụng tối đa kiến trúc hàng nghìn nhân tính toán của GPU vì ràng buộc tuần tự $t-1 \to t$.
   - Attention chuyển đổi bài toán chuỗi thành các phép toán đại số tuyến tính (GEMM - General Matrix Multiply), cho phép tăng tốc độ huấn luyện lên hàng chục lần trên các tập dữ liệu quy mô hàng tỷ từ.

### 2.2. Attention tạo ra thách thức mới nào? (Attention is NOT Free)
1. **Rào cản bùng nổ chi phí tính toán và bộ nhớ $\mathcal{O}(T^2)$ (Quadratic Bottleneck):**
   - Kích thước của ma trận Attention là $T \times T$. Khi độ dài văn bản tăng gấp $10$ lần ($1,000 \to 10,000$ tokens), số phép tính và dung lượng VRAM GPU tăng gấp $100$ lần ($10^2$).
   - Với văn bản dài (như sách, tài liệu pháp lý, mã nguồn), việc tính toán full Self-Attention trở nên bất khả thi nếu không có các kỹ thuật xấp xỉ tối ưu (FlashAttention, Sparse Attention, Longformer).
2. **Mất mát hoàn toàn khái niệm Trật tự từ (Permutation Invariance):**
   - Nếu hoán vị ngẫu nhiên vị trí các từ trong câu mà không có cơ chế bù đắp, ma trận Self-Attention chỉ đơn thuần hoán vị các hàng và cột tương ứng, context vector tổng thu được không đổi.
   - Do đó, Attention bắt buộc phải kết hợp với **Positional Encoding** (Mã hóa vị trí) để mô hình có thể phân biệt được ngữ nghĩa theo trật tự cú pháp.
3. **Ảo tưởng về tính giải thích (The Interpretability Trap):**
   - Người phát triển dễ ngộ nhận rằng trọng số Attention cao đồng nghĩa với việc mô hình đang "tư duy" logic như con người. Thực tế, Attention chỉ phản ánh tương quan tích vô hướng trong không gian ẩn, không đảm bảo tính nhân quả đối với quyết định cuối cùng.

---

## 3. Lời giải 9 câu hỏi kiểm tra năng lực cá nhân (Individual Learning Check)

### Câu 1: Attention score được tính như thế nào?
- **Trả lời:** Attention score $s_i$ đo lường độ tương đồng ngữ nghĩa giữa một vector truy vấn $q$ (Query) và một vector khóa $k_i$ (Key). Phương pháp phổ biến nhất là tích vô hướng (Dot-product):
  $$s_i = q^T k_i = \sum_{d=1}^{d_k} q_d \cdot k_{id}$$
  Trong mô hình Transformer, điểm số này được chuẩn hóa chia tỉ lệ: $s_i = \frac{q^T k_i}{\sqrt{d_k}}$.

### Câu 2: Tại sao phải dùng hàm Softmax để tính trọng số?
- **Trả lời:** Softmax đảm bảo hai tính chất toán học bắt buộc đối với một phân phối xác suất:
  1. **Tính không âm:** $\alpha_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)} > 0, \quad \forall i$.
  2. **Tính chuẩn hóa đơn vị:** $\sum_{i=1}^T \alpha_i = 1$.
  Nhờ đó, context vector $c = \sum \alpha_i v_i$ là một tổ hợp lồi (Convex Combination), đảm bảo độ lớn (norm) của vector đầu ra luôn nằm trong một miền giá trị ổn định, không bị bùng nổ biên độ. Ngoài ra, tính chất hàm mũ của Softmax giúp khuếch đại các giá trị lớn và dồn trọng số tập trung vào các token quan trọng nhất.

### Câu 3: Query, Key và Value có vai trò gì trong cơ chế Attention?
- **Trả lời:** Cơ chế này được kế thừa từ hệ thống truy xuất thông tin (Information Retrieval):
  - **Query ($Q$):** Đại diện cho đối tượng đang tìm kiếm thông tin ("Tôi đang quan tâm đến điều gì?").
  - **Key ($K$):** Đóng vai trò là nhãn nhận diện hoặc thuộc tính của các đối tượng trong bộ nhớ ("Tôi có đặc điểm gì để khớp với Query?").
  - **Value ($V$):** Chứa đựng nội dung thông tin thực sự sẽ được trích xuất nếu Key khớp với Query ("Nội dung thông tin mà tôi đóng góp là gì?").

### Câu 4: Context vector được tính như thế nào?
- **Trả lời:** Context vector $c$ được tính bằng tổng có trọng số của tất cả các vector Value $v_i$, trong đó trọng số chính là các giá trị Attention weights $\alpha_i$:
  $$c = \sum_{i=1}^T \alpha_i v_i = \alpha_1 v_1 + \alpha_2 v_2 + \dots + \alpha_T v_T$$
  Dưới dạng ma trận: $C = A V$ với $A \in \mathbb{R}^{T \times T}$ và $V \in \mathbb{R}^{T \times d_v}$.

### Câu 5: Tại sao Scaled Dot-Product Attention phải chia cho $\sqrt{d_k}$?
- **Trả lời:** Khi chiều không gian vector $d_k$ lớn, tích vô hướng $q^T k = \sum_{i=1}^{d_k} q_i k_i$ có phương sai tăng tuyến tính bằng $d_k$ (nếu các thành phần có kỳ vọng 0 và phương sai 1). Điều này khiến các giá trị score đạt độ lớn tuyệt đối rất cao, đẩy hàm Softmax vào vùng bão hòa có độ dốc cực nhỏ. Chia cho $\sqrt{d_k}$ giúp kéo phương sai của score về lại bằng $1$, duy trì gradient dồi dào cho thuật toán Backpropagation và tránh hiện tượng bão hòa gradient.

### Câu 6: Self-Attention khác Attention thông thường ở điểm nào?
- **Trả lời:** 
  - Trong **Attention thông thường** (như cơ chế Seq2Seq của Bahdanau), Query đến từ trạng thái ẩn của bộ giải mã (Decoder), trong khi Key và Value đến từ bộ mã hóa (Encoder) ở hai nguồn văn bản khác nhau.
  - Trong **Self-Attention** (Tự chú ý), cả $Q, K, V$ đều được sinh ra từ **cùng một chuỗi đầu vào** $X$ thông qua các phép chiếu tuyến tính: $Q = X W_Q, K = X W_K, V = X W_V$. Nhờ đó, các token trong cùng một câu có thể trực tiếp tương tác, đối chiếu và xây dựng biểu diễn ngữ cảnh lẫn nhau.

### Câu 7: Nếu độ dài chuỗi tăng gấp đôi ($T \to 2T$) thì ma trận Attention thay đổi như thế nào?
- **Trả lời:** Kích thước ma trận Attention tăng từ $T \times T$ lên $(2T) \times (2T) = 4 T^2$. Tức là số lượng phần tử cần tính toán và dung lượng bộ nhớ lưu trữ tăng **gấp 4 lần** (Tăng trưởng theo cấp số nhân bậc hai $\mathcal{O}(T^2)$).

### Câu 8: Trọng số Attention có chứng minh mô hình đang thực hiện suy luận logic (Reasoning) như con người không?
- **Trả lời:** **Không.** Các nghiên cứu kinh điển về khả giải thích của mô hình ngôn ngữ (Jain & Wallace 2019, Serrano & Smith 2019) đã chỉ ra rằng các phân phối Attention khác nhau hoàn toàn có thể dẫn đến cùng một kết quả dự đoán; và việc hoán đổi hay làm nhiễu Attention không làm suy giảm độ chính xác của mạng. Trọng số Attention chỉ là giá trị tương quan số học trong không gian ẩn của lớp đó, không phải là bằng chứng của quá trình lập luận ngữ nghĩa hay nhân quả.

### Câu 9: Tại sao Attention có thể xử lý tương tác tầm xa tốt hơn trạng thái hồi quy (Recurrent State)?
- **Trả lời:** Trong mạng hồi quy, thông tin từ token đầu muốn đến token cuối phải đi qua chuỗi liên tiếp $T-1$ bước nén phi tuyến, khiến tín hiệu bị pha loãng và gradient bị tiêu biến. Trong Self-Attention, khoảng cách đường truyền giữa hai token bất kỳ luôn bằng **$1$ bước tính toán ($\mathcal{O}(1)$)**. Bất kể khoảng cách giữa hai từ là 5 từ hay 500 từ, điểm tương đồng $q_i^T k_j$ vẫn được tính toán trực tiếp mà không phải chịu bất kỳ sự hao mòn thông tin nào dọc đường truyền.

---

## 4. Tuyên bố sử dụng AI (AI Assistance Statement)

Căn cứ theo quy định tại **Mục 30. AI Policy**, bản tuyên bố dưới đây xác định rõ phạm vi hỗ trợ của công cụ trí tuệ nhân tạo:

```yaml
AI Assistance Statement:
  Tool: Antigravity IDE (DeepMind Agentic Assistant)
  Purpose: 
    - Hỗ trợ xây dựng cấu trúc script tiền xử lý dữ liệu và vẽ biểu đồ Matplotlib độ phân giải cao.
    - Hỗ trợ rà soát tính ổn định số học của hàm Softmax trong NumPy.
    - Định dạng trực quan bảng Markdown và kiểm tra tính hợp lệ của notebook JSON.
  Generated:
    - Khung mã nguồn script vẽ đồ thị Matplotlib (generate_w6_plots.py).
    - Mẫu bảng tổng hợp số liệu thực nghiệm.
  Modified:
    - Loại bỏ hoàn toàn các comment code trong toàn bộ mã nguồn theo quy chuẩn nghiêm ngặt.
    - Tối ưu hóa thuật toán Stable Softmax trừ max(x) để ngăn chặn hoàn toàn lỗi tràn số float32/float64.
  Verified:
    - Sinh viên trực tiếp thực hiện tính toán bằng tay và đối chuẩn từng bước giải tích tại calculations.md.
    - Sinh viên tự biện luận các giả thuyết tại prediction.md và các ca phân tích định tính tại error_analysis.md.
    - Khớp chính xác 100% giữa kết quả tính tay lý thuyết và kết quả chạy code thực tế.
```

### Năng lực giải thích các thành phần cốt lõi của sinh viên:
- **$Q$ (Query):** $Q = X W_Q$, không gian biểu diễn câu hỏi trích xuất đặc trưng.
- **$K$ (Key):** $K = X W_K$, không gian đánh chỉ mục để so khớp với câu hỏi.
- **$V$ (Value):** $V = X W_V$, không gian lưu trữ nội dung thông tin thực tế.
- **$Q K^T$:** Ma trận điểm tương tác đo mức độ tương đồng giữa mọi cặp token trong chuỗi.
- **$\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$:** Chuẩn hóa xác suất phân phối mức độ tập trung trên từng hàng.
- **$A V$:** Tổ hợp tuyến tính trích xuất vector biểu diễn ngữ cảnh hoàn chỉnh.
