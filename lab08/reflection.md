# Reflection

> **AI usage:** Bài reflection và phần trả lời Individual Learning Check trong file này do AI (Claude, Anthropic) soạn dựa trên kết quả thực nghiệm của LAB 08.

## Reflection

> Sau LAB 07, chúng ta đã biết cách xây dựng Transformer. Sau LAB 08, tại sao trong thực tế người ta thường không bắt đầu bằng việc train Transformer từ đầu?

LAB 07 cho thấy Transformer chỉ là các khối quen thuộc ghép lại. Nhưng biết lắp ráp kiến trúc khác với có đủ điều kiện để huấn luyện nó tốt từ đầu.

Thứ nhất là **dữ liệu**. Một Transformer khởi tạo ngẫu nhiên phải học mọi thứ từ dữ liệu của bài toán: nghĩa của từ, ngữ pháp, cách phủ định đảo ngược cảm xúc, và cuối cùng mới đến ranh giới positive/negative. Với 4000 review, model from scratch chỉ đạt 0.688 accuracy; với 400 review nó gần như đoán mò (0.588). Bài toán thực tế thường chỉ có vài nghìn mẫu gán nhãn.

Thứ hai là **pretrained representation**. DistilBERT đã học biểu diễn ngôn ngữ từ kho văn bản lớn bằng Masked Language Modeling, nên vector `[CLS]` của nó đã mang thông tin hữu ích trước khi thấy một review IMDb nào. Bằng chứng rõ nhất là chế độ frozen: chỉ train 1,538 tham số của lớp tuyến tính mà vẫn đạt 0.800, vượt model from scratch được train toàn bộ. Pretrained chỉ cần 25% dữ liệu (0.805) để vượt xa model from scratch dùng 100% dữ liệu.

Thứ ba là **chi phí tính toán**. Fine-tune DistilBERT mất 88 giây trên một GPU T4, còn pretrain chính model đó đòi hỏi hàng trăm GPU-giờ trên hàng tỷ token. Chi phí vẫn phải cân nhắc khi fine-tune: tăng T từ 128 lên 256 cải thiện accuracy từ 0.844 lên 0.891 nhưng thời gian train và bộ nhớ GPU đều tăng gấp đôi, đúng với bản chất $\mathcal{O}(T^2)$ của attention.

Thứ tư là **downstream task** và **fine-tuning**. Pretraining không nhắm đến phân loại cảm xúc; nó học ngôn ngữ nói chung. Classification head là phần duy nhất gắn với nhiệm vụ, còn fine-tuning điều chỉnh toàn bộ encoder để biểu diễn tập trung vào tín hiệu cần thiết. Trong thí nghiệm, fine-tuning tốt hơn frozen 4.4 điểm: encoder pretrained đã khá phù hợp, fine-tuning bổ sung phần thích nghi cuối cùng. Cùng một encoder có thể dùng lại cho nhiều nhiệm vụ khác nhau chỉ bằng cách đổi head, như LAB 09 sẽ làm với NER.

Cuối cùng là **hạn chế và domain shift**. Pretrained không có nghĩa là hiểu mọi domain. Khi test trên tweets, accuracy giảm từ 0.844 xuống 0.771 và precision chỉ còn 0.656. Error analysis cũng cho thấy model sai với confidence trên 0.97 ở các review châm biếm hoặc có ý chính nằm sau phần bị cắt. Model pretrained giảm nhu cầu dữ liệu, nhưng vẫn cần đánh giá trên đúng domain sử dụng, chọn sequence length phù hợp với dữ liệu, và không được coi confidence là xác suất đúng.

Vì vậy, người ta bắt đầu từ model pretrained rồi fine-tune: rẻ hơn, cần ít dữ liệu hơn và tốt hơn nhiều so với train từ đầu, miễn là nhận thức rõ giới hạn của nó.

## Individual Learning Check

1. **Pretraining khác fine-tuning ở đâu?** Pretraining huấn luyện model trên kho văn bản lớn, không gán nhãn, với mục tiêu tổng quát như MLM để học biểu diễn ngôn ngữ. Fine-tuning bắt đầu từ trọng số pretrained và tiếp tục train trên dữ liệu có nhãn của một nhiệm vụ cụ thể, với loss của nhiệm vụ đó.
2. **MLM là gì?** Masked Language Modeling: che ngẫu nhiên một số token trong câu bằng `[MASK]` và bắt model dự đoán token gốc từ ngữ cảnh hai phía, tối ưu bằng cross-entropy trên các vị trí bị che.
3. **Tại sao cần tokenizer?** Model chỉ nhận số. Tokenizer chuyển văn bản thành chuỗi token và token ID theo đúng vocabulary mà model đã được pretrain, kèm special tokens và attention mask.
4. **Subword token giải quyết vấn đề gì?** Giữ vocabulary ở kích thước hữu hạn (khoảng 30k với DistilBERT) mà vẫn biểu diễn được từ hiếm và từ chưa gặp bằng cách ghép các mảnh nhỏ (các mảnh nối tiếp có tiền tố `##`), nên hầu như không phải dùng `[UNK]`.
5. **`[CLS]` thường được sử dụng như thế nào?** Được đặt ở đầu chuỗi; nhờ self-attention, hidden state cuối cùng của nó tổng hợp thông tin cả chuỗi và được đưa vào classification head để phân loại câu.
6. **Attention mask dùng để làm gì?** Đánh dấu token thật (1) và token padding (0) để attention bỏ qua `[PAD]`, nên padding không ảnh hưởng tới biểu diễn của câu.
7. **Frozen encoder nghĩa là gì?** Giữ nguyên toàn bộ trọng số encoder pretrained (không cập nhật gradient), chỉ train classification head trên đặc trưng mà encoder sinh ra; đây là feature extraction.
8. **Fine-tuning thay đổi những tham số nào?** Full fine-tuning cập nhật tất cả tham số: embedding, các layer attention và feed-forward của encoder, cùng classification head (66.4M tham số với DistilBERT trong lab này).
9. **Vì sao pretrained model có thể hữu ích khi downstream dataset nhỏ?** Model đã có sẵn biểu diễn ngôn ngữ nên dữ liệu nhỏ chỉ cần dùng để học ranh giới nhiệm vụ. Trong lab, với 400 review, pretrained đạt 0.700 so với 0.588 của from scratch.
10. **Pretrained model có đảm bảo không bị domain shift không?** Không. Train trên IMDb rồi test trên tweets, accuracy giảm từ 0.844 xuống 0.771 do văn phong, từ vựng và phân phối nhãn khác nhau.
11. **Vì sao tăng sequence length làm chi phí attention tăng?** Self-attention tính điểm giữa mọi cặp token, nên ma trận attention có kích thước T × T: chi phí tính toán và bộ nhớ tăng theo $\mathcal{O}(T^2)$. Trong lab, T = 256 tốn khoảng gấp đôi thời gian và bộ nhớ so với T = 128.
12. **Confidence cao có đảm bảo prediction đúng không?** Không. Nhóm confidence ≥ 0.9 vẫn có 58 dự đoán sai, và các ví dụ lỗi tiêu biểu đều có confidence trên 0.97.
