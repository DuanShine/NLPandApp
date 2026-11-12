# Error Analysis

> **AI usage:** Phần phân loại lỗi, giả thuyết, confidence analysis và trả lời Q1–Q6 trong file này do AI (Claude, Anthropic) soạn dựa trên kết quả trong `results.csv` và `fine_tuning.ipynb`.

Model: DistilBERT fine-tuned (T = 128, 100% data), IMDb test (2000 reviews).
Confusion matrix: TN = 826, FP = 174, FN = 138, TP = 862.

Các ví dụ dưới đây được chọn tự động trong `fine_tuning.ipynb` (`pick_errors`): 2 ví dụ sai với confidence cao nhất, sau đó 3 false positives và 3 false negatives có confidence cao nhất tiếp theo. Văn bản được cắt ở 600 ký tự; đầy đủ trong notebook.

Error type: negation / sarcasm / long-range dependency / ambiguous sentiment / domain-specific expression / rare word / tokenization / context / label ambiguity.

| # | Selection | Text | True | Pred | Confidence | Error type | Hypothesis |
|---|---|---|---|---|---|---|---|
| 1 | very confident & wrong | I've been a fan of Jim Henson and his characters since the very beginning. The most beguiling thing about them was the love and innocence and camaraderie shown. Kermit was a role model of deep thinking and problem solving. A spiritual character, yet sweetly and believably so. All the other characters were slightly eccentric but it demonstrated how different kinds of beings can co-exist in a caring manner together, respecting each other's difference.  Following movies have somewhat kept the same vibe. Yet this "It's a Very Merry Muppet Christmas Movie" would have Jim Henson spinning in his grav ... | negative | positive | 0.983 | long-range dependency + truncation | 120 từ đầu ca ngợi Jim Henson và các nhân vật gốc; lời chê phim mới chỉ xuất hiện sau đó. Với T = 128, model gần như chỉ thấy phần khen. |
| 2 | very confident & wrong | Each year the company called Nu produces couple of "action packed", "full of suspense" movies. This little nugget, called Shadowcaster III(Until I visited this site I wasn't avare this is a whole trilogy), is a great example of the good job the company is doing. Frank Zagarino is as mean as always and does a great job as almost undestructable(?), schizophrenic(??) android. I won't waste any more words since I don't want to reveal the terrific plot and ruin you a couple of great laughs.  Rating 2/10 (Revard for those hard working tehnicians. Man, putting together this kind of rubbish must be ne ... | negative | positive | 0.981 | sarcasm | Cả review khen mỉa ("great example of the good job", "terrific plot"); tín hiệu thật duy nhất là "Rating 2/10" và "rubbish" ở cuối, có thể đã bị cắt. |
| 3 | false positive | Well, for a start, I must say that, here, in Russia, a saga of Geralt of Rivia is known and loved. Andrzej Sapkowski - a gifted writer, or, maybe, even genius of some sort - created one of the most realistic, honest, cynical and God d@mn well written fantasy worlds in a history of literature. And when such amazing material gets in a right hands well, see RPG game "Witcher" and you will understand what I am talking about.  "Vedmak\Wiedzmin" is an excellent example of the opposite outcome.  Lack of budget. Lack of directing. Lack of good script.  These three whales of Disgusting Movie Making sun ... | negative | positive | 0.981 | context + truncation | Phần đầu khen tiểu thuyết của Sapkowski và game Witcher (thực thể khác bộ phim); lời chê phim ("Lack of budget... Disgusting") nằm sau. Model gán cảm xúc của thực thể được nhắc trước cho cả review. |
| 4 | false positive | Earnest effort which achieves some success to adapt the classic Odyssey story to a '30's nostalgia period piece. The adventurers this time are escaped convicts, wandering about the Depression afflicted South bungling their way into trouble.  The greatest strength of the film is the wonderful music soundtrack, effectively evoking not only the era that this is set in, but the spiritual references that run rampant in the film. Besides its value to the tone of the movie, the music is just plain fun to listen to.  What cripples the film is that the characters really aren't that likable. In a comedy ... | negative | positive | 0.981 | ambiguous sentiment + truncation | Mở đầu bằng lời khen có chừng mực ("earnest effort", "wonderful music soundtrack"); phần chê nhân vật, chiếm phần lớn review (363 từ), nằm sau 128 token đầu. |
| 5 | false positive | If you never have read the book and never intend to read it in the future, go on and watch the movie (6/10). It is a nice fantasy movie with well done CGI, nice acting, a beautiful environment and an above-average fantasy story.  If you have read the book like me about 10 times or more and really love it, don't expect too much (or better: don't expect anything at all). The story is totally different from the original book. This may explain that the movie is voted 1/10 from people around 40 or more (like me) and much better from people who most probably never read the book before and thus expec ... | negative | positive | 0.977 | ambiguous sentiment / label ambiguity | Câu đầu tự cho 6/10 và khen CGI, diễn xuất; phần còn lại so sánh với sách và chê. Review có hai nhóm người xem với hai đánh giá ngược nhau, nhãn negative khá mơ hồ. |
| 6 | false negative | You have to be awfully patient to sit through a film with one-liners so flat and unfunny that you wonder what all the fuss was about when WHISTLING IN THE DARK opened to such an enthusiastic greeting from audiences in the 1940s.  On top of some weak one-liners and ordinary sight gags, the plot is as far-fetched as the tales The Fox (Red Skelton) tells his radio audience. You have to wonder why anyone would think he could come up with a real-life solution on how to commit the perfect crime and get away with it. But then, that's how unrealistic the comedy is.  But--if you're a true Red Skelton f ... | positive | negative | 0.976 | label ambiguity + long-range dependency | Hai đoạn đầu chê thẳng ("flat and unfunny", "far-fetched"); phần dễ chịu hơn chỉ dành cho fan Red Skelton ở cuối. Nội dung nghiêng về trung tính/tiêu cực dù nhãn là positive. |
| 7 | false negative | This is not the kind of movie that really merits critical attention. It's not going to win any Oscars - it's really not a very good movie at all. Heck, it's not even really a movie - just a string of short disgusting, gross-out skits strung together in an effort to make entertainment. And as much as every critical bone in my body cries out to give this movie a failing score, gosh darn it, I'm going to give it a much better score. The kind of people who are going to enjoy this movie are not the kind of people who really care what critics have to say, so I'm not going to give some snobby critiqu ... | positive | negative | 0.973 | negation / contrast | Dày đặc từ tiêu cực ("not a very good movie", "gross-out crap", "disgusting") nhưng kết luận lại là "It's actually pretty funny", "entertaining". Model bám vào từ phủ định/tiêu cực thay vì nắm được kết luận đảo chiều. |
| 8 | false negative | Every episode I saw when I was an innocent child was stupid. There were some funny looking puppets called "The Mitts" who would always discuss childhood things. They would play this song before "The Mitts" part of the show that goes like this "Let's join the Mitts..." There was a Groucho Marx puppet that told jokes. Before the session, this man would sing "Hot Fudge! Right On!" There was a green puppet with teeth that was called Seymore. We would always see more Seymore. He would crack jokes and sing every episode. There was always a moral in every episode. One episode, a man sang "Liars are l ... | positive | negative | 0.973 | label ambiguity + rare word | Câu đầu "was stupid", phần còn lại chỉ mô tả hoài niệm (tên con rối, bài hát), không có lời khen rõ ràng. Nhãn positive khó suy ra từ văn bản; tên riêng hiếm ("Seymore", "Mitts") bị tách thành nhiều subword. |

## Tổng hợp quan sát từ 8 ví dụ

- **Truncation là nguyên nhân chung lớn nhất:** ở 6/8 ví dụ (#1, #3, #4, #5, #6, #7), ý kiến chính của người viết nằm sau khoảng 128 token đầu. Độ dài median của review IMDb test là 227 token, nên 88.3% review bị cắt ở T = 128. Bằng chứng hỗ trợ: tăng T lên 256 nâng accuracy từ 0.844 lên 0.891 (Exp 4).
- **Mở đầu khen, kết thúc chê** (#1, #3, #4, #5) dẫn tới false positive; **mở đầu chê, kết luận khen** (#6, #7) dẫn tới false negative. Model dựa nhiều vào giọng điệu của phần đầu.
- **Sarcasm** (#2) và **label ambiguity** (#5, #6, #8) là những lỗi mà tăng sequence length chưa chắc sửa được.
- Cả 8 ví dụ đều sai với confidence ≥ 0.97, tức là các lỗi này không thể lọc bằng ngưỡng confidence.

## Confidence analysis

| Confidence bin | n | Accuracy | Số dự đoán sai |
|---|---|---|---|
| Low (0.5–0.7) | 285 | 0.5018 | 142 |
| Medium (0.7–0.9) | 463 | 0.7581 | 112 |
| High (≥ 0.9) | 1252 | 0.9537 | 58 |

**Confidence cao có đồng nghĩa với prediction đúng không?** Không.

- *Observation:* accuracy tăng dần theo nhóm confidence (0.50 → 0.76 → 0.95), nhưng nhóm High vẫn có 58 dự đoán sai, và cả 8 ví dụ lỗi ở trên đều có confidence từ 0.973 đến 0.983.
- *Evidence:* trên tweets (out-of-domain), tỉ lệ mẫu ở nhóm Low tăng từ 14% lên 46%, cho thấy confidence có phản ánh phần nào sự không chắc chắn; nhưng các lỗi do truncation và sarcasm vẫn cho confidence gần 1.
- *Interpretation:* softmax confidence là mức tự tin của model với phần văn bản nó nhìn thấy, không phải xác suất đúng. Khi phần quyết định cảm xúc bị cắt hoặc bị nói ngược, model vẫn rất chắc chắn với phần còn lại. Confidence hữu ích để xếp hạng (nhóm Low nên được kiểm tra lại), nhưng không thay được việc đánh giá đúng/sai: confidence ≠ correctness.

## Câu hỏi bắt buộc

### Q1. Model nào hưởng lợi nhiều hơn khi training data ít?

- *Observation:* ở 10% dữ liệu (400 reviews), pretrained đạt accuracy 0.700 / F1 0.723, còn from scratch đạt 0.588 / 0.577. Pretrained cao hơn ở mọi mức dữ liệu (khoảng cách 11.2, 17.9, 19.7 và 15.7 điểm ở 10/25/50/100%).
- *Evidence:* pretrained chỉ cần 25% dữ liệu (0.805) để vượt xa scratch dùng 100% dữ liệu (0.688). Đường cong pretrained gần như bão hòa từ 50% (0.843 so với 0.844 ở 100%), trong khi scratch vẫn tăng đều và chưa bão hòa.
- *Interpretation:* model pretrained hưởng lợi nhiều hơn khi dữ liệu ít: nó đã có biểu diễn ngôn ngữ từ pretraining nên chỉ cần ít mẫu để học ranh giới cảm xúc. Lưu ý mức giảm tuyệt đối từ 100% xuống 10% của pretrained (14.5 điểm) lớn hơn của scratch (10 điểm), vì scratch ở 10% đã gần mức đoán ngẫu nhiên (0.5) nên không còn nhiều chỗ để giảm. Vì vậy phải nhìn cả đường cong, không chỉ một con số.

### Q2. Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?

- *Observation:* fine-tuned DistilBERT đạt 0.844 so với 0.688 của Transformer from scratch trên cùng dữ liệu, split, T = 128 và seed.
- *Evidence:* hai model khác nhau ở **cả** trọng số khởi tạo (pretrained vs ngẫu nhiên) **lẫn** kích thước (66.4M vs 4.2M tham số, 6 vs 2 layers) và số epoch (2 vs 5).
- *Interpretation:* kết quả cho thấy pretrained + fine-tuning hiệu quả hơn hẳn khi dữ liệu ít (4000 reviews), nhưng **không chứng minh riêng** rằng pretraining là nguyên nhân duy nhất, vì kích thước model cũng khác. Thí nghiệm kiểm soát chặt hơn sẽ dùng cùng kiến trúc DistilBERT nhưng khởi tạo ngẫu nhiên. Kết quả cũng không chứng minh model "hiểu ngôn ngữ"; nó chỉ cho thấy biểu diễn học từ kho văn bản lớn có ích cho bài toán sentiment này.

### Q3. Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?

- *Observation:* frozen đạt 0.800 / F1 0.796, kém fine-tuning 4.4 điểm accuracy, trong khi chỉ train 1,538 tham số (so với 66.4M) và mất 20.4 s thay vì 88.1 s.
- *Evidence:* một bộ phân loại tuyến tính trên vector `[CLS]` cố định đã đạt 80%, cao hơn hẳn model from scratch (0.688) được train toàn bộ.
- *Interpretation:* biểu diễn `[CLS]` có sẵn từ pretraining đã chứa phần lớn thông tin cảm xúc, gần như tách được bằng một siêu phẳng. Nghĩa là phần lớn "kiến thức" nằm trong encoder pretrained; khi tài nguyên hạn chế, feature extraction là lựa chọn rẻ mà vẫn đạt kết quả khá.

### Q4. Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?

- *Observation:* trong thí nghiệm này khoảng cách là 4.4 điểm accuracy (0.844 vs 0.800) và 5.1 điểm F1: có ý nghĩa nhưng không quá lớn.
- *Evidence:* `[CLS]` của DistilBERT được pretrain bằng MLM, không được huấn luyện để tóm tắt cảm xúc cả câu; fine-tuning cho phép mọi layer điều chỉnh để `[CLS]` tập trung vào tín hiệu cảm xúc.
- *Interpretation:* nếu khoảng cách rất lớn, điều đó gợi ý biểu diễn pretrained chưa khớp với nhiệm vụ hoặc domain, nên cần thích nghi sâu ở các layer bên trong chứ không chỉ thiếu một classification head tốt. Ở đây khoảng cách vừa phải cho thấy biểu diễn pretrained đã khá phù hợp, fine-tuning chủ yếu tinh chỉnh thêm.

### Q5. Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?

- *Observation:* train trên IMDb, model fine-tuned giảm từ 0.844 (IMDb) xuống 0.771 (tweets), F1 từ 0.847 xuống 0.732; model from scratch giảm từ 0.688 xuống 0.463, tệ hơn đoán ngẫu nhiên.
- *Evidence:* precision trên tweets chỉ 0.656 (nhiều false positive) trong khi recall là 0.828. Tweets ngắn, nhiều tiếng lóng, hashtag, @mention, emoji và có tỉ lệ nhãn khác (37.7% positive so với 50% ở IMDb). Model from scratch chỉ từng thấy từ vựng và văn phong của review phim nên sụp đổ.
- *Interpretation:* các nguyên nhân có thể: (1) khác biệt văn phong và độ dài (review dài, trang trọng vs tweet ngắn, thân mật); (2) từ ngữ chỉ có ở domain mới bị tokenizer tách vụn; (3) cách biểu đạt cảm xúc khác (sarcasm, emoji); (4) lệch phân phối nhãn khiến model thiên về dự đoán positive. Pretraining giúp giảm mức sụt (fine-tuned và frozen vẫn giữ được 0.76–0.77) nhưng không loại bỏ domain shift: pretrained không có nghĩa là "hiểu mọi domain".

### Q6. Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?

- *Observation:* trong thí nghiệm này performance **không** đứng yên: T = 64 → 128 → 256 cho accuracy 0.792 → 0.844 → 0.891, nhưng training time tăng 47 → 88 → 182 s và bộ nhớ GPU ở T = 256 lên 4.7 GB.
- *Evidence:* 41.5% review vẫn dài hơn 256 token và 12.8% dài hơn 512 token; error analysis cho thấy 6/8 lỗi do ý chính nằm ngoài 128 token đầu. Chi phí attention tăng theo $\mathcal{O}(T^2)$.
- *Interpretation:* nếu performance gần như không đổi trong khi chi phí tăng mạnh thì không nên tăng tiếp: phần văn bản thêm vào không mang thêm thông tin cho nhiệm vụ mà chỉ tốn chi phí bậc hai. Quyết định nên dựa trên phân bố độ dài dữ liệu và đường cong performance/chi phí. Với IMDb, tăng lên 256 rõ ràng đáng giá; lên 512 (giới hạn của DistilBERT) có thể còn cải thiện nhưng chi phí attention tăng khoảng 4 lần so với 256, cần thử nghiệm mới kết luận được.
