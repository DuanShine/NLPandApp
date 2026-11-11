# Error Analysis

Model: DistilBERT fine-tuned (T = 128, 100% data), IMDb test (2000 reviews).
Confusion matrix: TN = 826, FP = 174, FN = 138, TP = 862.

Các ví dụ dưới đây được chọn tự động trong `fine_tuning.ipynb` (`pick_errors`): 2 ví dụ sai với confidence cao nhất, sau đó 3 false positives và 3 false negatives có confidence cao nhất tiếp theo. Văn bản được cắt ở 600 ký tự; đầy đủ trong notebook.

Error type: negation / sarcasm / long-range dependency / ambiguous sentiment / domain-specific expression / rare word / tokenization / context / label ambiguity.

| # | Selection | Text | True | Pred | Confidence | Error type | Hypothesis |
|---|---|---|---|---|---|---|---|
| 1 | very confident & wrong | I've been a fan of Jim Henson and his characters since the very beginning. The most beguiling thing about them was the love and innocence and camaraderie shown. Kermit was a role model of deep thinking and problem solving. A spiritual character, yet sweetly and believably so. All the other characters were slightly eccentric but it demonstrated how different kinds of beings can co-exist in a caring manner together, respecting each other's difference.  Following movies have somewhat kept the same vibe. Yet this "It's a Very Merry Muppet Christmas Movie" would have Jim Henson spinning in his grav ... | negative | positive | 0.983 |  |  |
| 2 | very confident & wrong | Each year the company called Nu produces couple of "action packed", "full of suspense" movies. This little nugget, called Shadowcaster III(Until I visited this site I wasn't avare this is a whole trilogy), is a great example of the good job the company is doing. Frank Zagarino is as mean as always and does a great job as almost undestructable(?), schizophrenic(??) android. I won't waste any more words since I don't want to reveal the terrific plot and ruin you a couple of great laughs.  Rating 2/10 (Revard for those hard working tehnicians. Man, putting together this kind of rubbish must be ne ... | negative | positive | 0.981 |  |  |
| 3 | false positive | Well, for a start, I must say that, here, in Russia, a saga of Geralt of Rivia is known and loved. Andrzej Sapkowski - a gifted writer, or, maybe, even genius of some sort - created one of the most realistic, honest, cynical and God d@mn well written fantasy worlds in a history of literature. And when such amazing material gets in a right hands well, see RPG game "Witcher" and you will understand what I am talking about.  "Vedmak\Wiedzmin" is an excellent example of the opposite outcome.  Lack of budget. Lack of directing. Lack of good script.  These three whales of Disgusting Movie Making sun ... | negative | positive | 0.981 |  |  |
| 4 | false positive | Earnest effort which achieves some success to adapt the classic Odyssey story to a '30's nostalgia period piece. The adventurers this time are escaped convicts, wandering about the Depression afflicted South bungling their way into trouble.  The greatest strength of the film is the wonderful music soundtrack, effectively evoking not only the era that this is set in, but the spiritual references that run rampant in the film. Besides its value to the tone of the movie, the music is just plain fun to listen to.  What cripples the film is that the characters really aren't that likable. In a comedy ... | negative | positive | 0.981 |  |  |
| 5 | false positive | If you never have read the book and never intend to read it in the future, go on and watch the movie (6/10). It is a nice fantasy movie with well done CGI, nice acting, a beautiful environment and an above-average fantasy story.  If you have read the book like me about 10 times or more and really love it, don't expect too much (or better: don't expect anything at all). The story is totally different from the original book. This may explain that the movie is voted 1/10 from people around 40 or more (like me) and much better from people who most probably never read the book before and thus expec ... | negative | positive | 0.977 |  |  |
| 6 | false negative | You have to be awfully patient to sit through a film with one-liners so flat and unfunny that you wonder what all the fuss was about when WHISTLING IN THE DARK opened to such an enthusiastic greeting from audiences in the 1940s.  On top of some weak one-liners and ordinary sight gags, the plot is as far-fetched as the tales The Fox (Red Skelton) tells his radio audience. You have to wonder why anyone would think he could come up with a real-life solution on how to commit the perfect crime and get away with it. But then, that's how unrealistic the comedy is.  But--if you're a true Red Skelton f ... | positive | negative | 0.976 |  |  |
| 7 | false negative | This is not the kind of movie that really merits critical attention. It's not going to win any Oscars - it's really not a very good movie at all. Heck, it's not even really a movie - just a string of short disgusting, gross-out skits strung together in an effort to make entertainment. And as much as every critical bone in my body cries out to give this movie a failing score, gosh darn it, I'm going to give it a much better score. The kind of people who are going to enjoy this movie are not the kind of people who really care what critics have to say, so I'm not going to give some snobby critiqu ... | positive | negative | 0.973 |  |  |
| 8 | false negative | Every episode I saw when I was an innocent child was stupid. There were some funny looking puppets called "The Mitts" who would always discuss childhood things. They would play this song before "The Mitts" part of the show that goes like this "Let's join the Mitts..." There was a Groucho Marx puppet that told jokes. Before the session, this man would sing "Hot Fudge! Right On!" There was a green puppet with teeth that was called Seymore. We would always see more Seymore. He would crack jokes and sing every episode. There was always a moral in every episode. One episode, a man sang "Liars are l ... | positive | negative | 0.973 |  |  |

## Confidence analysis

| Confidence bin | n | Accuracy | Số dự đoán sai |
|---|---|---|---|
| Low (0.5–0.7) | 285 | 0.5018 | 142 |
| Medium (0.7–0.9) | 463 | 0.7581 | 112 |
| High (≥ 0.9) | 1252 | 0.9537 | 58 |

Confidence cao có đồng nghĩa với prediction đúng không?

_(Sinh viên tự viết)_

## Câu hỏi bắt buộc

Trả lời theo: observation → evidence → interpretation (số liệu: `README.md` mục 5, `results.csv`).

### Q1. Model nào hưởng lợi nhiều hơn khi training data ít?

### Q2. Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?

### Q3. Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?

### Q4. Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?

### Q5. Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?

### Q6. Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?
