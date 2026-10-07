# A. 3 Similarity Đúng

## 1. `medical` → `clinical`

* **Observed:** Cosine Similarity = **0.8067**
* **Expected:** Hai từ mang tính chất đồng nghĩa hoặc có ngữ cảnh gần như tương đương trong ngành y tế.
* **Possible explanation:** Trong corpus huấn luyện, `medical` và `clinical` xuất hiện ở các vị trí cú pháp và ngữ cảnh xung quanh giống nhau với tần suất lớn. Ví dụ: `medical research`, `clinical research`, `medical trial`, `clinical trial`.
* **Evidence from corpus:** Ma trận chỉ ra cặp `medical` và `clinical` đạt điểm số cao nhất toàn ma trận: **0.8067**.
* **Các nguyên nhân có thể:**

  * **Context window:** Cửa sổ ngữ cảnh bắt đúng các cụm từ ghép chuyên ngành.
  * **Frequency:** Tần suất xuất hiện đồng thời trong cùng chủ đề cao.

## 2. `treatment` → `therapy`

* **Observed:** Cosine Similarity = **0.6366**
* **Expected:** Hai từ thuộc cùng trường nghĩa về điều trị và trị liệu, có thể thay thế cho nhau trong ngữ cảnh y khoa.
* **Possible explanation:** Mô hình Word2Vec đã học được sự tương đồng về vị trí đứng của hai từ này khi miêu tả phương pháp chữa bệnh cho bệnh nhân.
* **Evidence from corpus:** Cặp `treatment` và `therapy` đạt **0.6366** trong ma trận so sánh từ với từ.
* **Các nguyên nhân có thể:**

  * **Context window:** Bao phủ được các cấu trúc câu chỉ hành động chữa bệnh.
  * **Frequency:** Xuất hiện nhiều trong ngữ cảnh y tế.

## 3. `medical` → `hospital`

* **Observed:** Cosine Similarity = **0.6014**
* **Expected:** Mối quan hệ giữa lĩnh vực `medical` và địa điểm hoặc cơ sở thực thi `hospital`.
* **Possible explanation:** Hai từ này có tần suất đồng xuất hiện trong cùng một cửa sổ ngữ cảnh rất cao ở các tài liệu tin tức và y học.
* **Evidence from corpus:** Điểm Cosine trong ma trận giữa `medical` và `hospital` đạt **0.6014**.
* **Các nguyên nhân có thể:**

  * **Context window:** Cửa sổ ngữ cảnh gom được các danh từ địa điểm và tính từ hoặc danh từ đi kèm.
  * **Domain bias:** Dữ liệu huấn luyện có mảng tin tức y tế và bệnh viện phong phú.

# B. 3 Similarity Sai / Bất ngờ

## 1. `medical` → `mechanical`

* **Observed:** Cosine Similarity = **0.5495**
* **Expected:** Điểm số phải rất thấp, gần **0.0**, vì `mechanical` không thuộc trường từ vựng y tế.
* **Possible explanation:**

  1. Corpus huấn luyện chứa các câu về trang thiết bị y tế như `mechanical ventilator`, `mechanical medical device`, làm kéo hai vector này lại gần nhau.
  2. Số lượng epoch huấn luyện hoặc dung lượng corpus chưa đủ để phân tách sắc thái giữa kỹ thuật y tế và y tế lâm sàng.
* **Evidence from corpus:** Đoạn log thực nghiệm trả về:

```text
Cosine Similarity ('medical', 'mechanical') = 0.5495
```

* **Các nguyên nhân có thể:**

  * **Noisy data:** Dữ liệu chứa các cụm từ kỹ thuật và thiết bị trộn lẫn trong văn bản y tế.
  * **Corpus nhỏ:** Dung lượng dữ liệu chưa đủ đại diện để phân biệt rõ ranh giới ngữ nghĩa.
  * **Insufficient training:** Số bước huấn luyện chưa đủ để tối ưu hóa không gian vector.

## 2. `medical` / `treatment` → `sick`

* **Observed:**

  * Cosine Similarity `medical` và `sick` = **0.0086**
  * Cosine Similarity `treatment` và `sick` = **-0.1292**
* **Expected:** Từ `sick` phải có liên quan ngữ nghĩa chặt chẽ với `medical` và `treatment`.
* **Possible explanation:** Từ `sick` là từ vựng thông dụng, trong khi `medical`, `treatment`, `clinical` thuộc văn phong báo chí và chuyên môn. Sự bất đồng về phong cách ngôn ngữ khiến ngữ cảnh xung quanh từ `sick` trong corpus huấn luyện khác biệt so với các từ chuyên ngành y tế.
* **Evidence from corpus:** Bảng ma trận ghi nhận:

  * `medical` | `sick` : **0.0086**
  * `treatment` | `sick` : **-0.1292**
* **Các nguyên nhân có thể:**

  * **Polysemy / Domain bias:** Từ `sick` xuất hiện trong nhiều ngữ cảnh đời sống thường ngày thay vì văn bản y khoa chính thống.
  * **Frequency:** Tần suất đồng xuất hiện trực tiếp giữa `sick` và `treatment` trong corpus huấn luyện rất thấp.

## 3. Sai lệch xếp hạng tìm kiếm: Query `medical treatment` chọn sai Top 1

* **Observed:** Query `medical treatment` trả về **Top 1** là câu Machine Learning với Score = **0.5428**, trong khi câu thực sự chứa thông tin y tế chỉ xếp **Top 2** với Score = **0.4166**.
* **Expected:** Câu y tế chứa các từ tương đồng cao như `clinical`, `therapy`, `hospital` phải xếp **Top 1** với điểm số vượt trội.
* **Possible explanation:**

  1. **Hạn chế của Mean Pooling:** Kỹ thuật lấy trung bình cộng làm xói mòn thông tin. Các từ phụ trong câu y tế như `the`, `sick`, `for`, `and` có điểm số rất thấp hoặc âm, làm kéo tụt trung bình cộng vector của cả câu.
  2. **Query Expansion bị nhiễu:** Bước mở rộng Query liên kết thêm các khái niệm nhiễu như `enforcement`, `regulatory`, `risk`, làm trôi dạt vector Query ban đầu về phía mảng công nghệ và dữ liệu.
* **Evidence from corpus:** Log tìm kiếm kết hợp ma trận chi tiết:

```text
Top 1  Score: 0.5428
machine learning algorithms require large amounts of data

Top 2  Score: 0.4166
the hospital provides clinical therapy and cares for the sick
```

Dù `medical` và `clinical` có Similarity = **0.8067**, việc tính trung bình trên toàn bộ câu khiến điểm của câu y tế bị kéo giảm xuống **0.4166**.

* **Các nguyên nhân có thể:**

  * **Vocabulary limitation:** Hạn chế của thuật toán nén câu bằng Mean Pooling. Cách tiếp cận này cần được thay thế bằng các mô hình biểu diễn câu như Sentence-Transformers hoặc BERT để chấm điểm ngữ cảnh toàn câu.
  * **Noisy data:** Sự hiện diện của các từ không mang ngữ nghĩa y tế kéo tụt chất lượng biểu diễn của câu.
