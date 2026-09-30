import math
from collections import Counter
import numpy as np

class NgramTextModel:
    def __init__(self, degree):
        self.degree = degree
        self.vocab_set = set()
        self.freq_uni = Counter()
        self.freq_bi = Counter()
        self.freq_tri = Counter()
        self.total_words = 0

    def _create_vocab(self, text_data):
        self.vocab_set = set()
        for seq in text_data:
            for w in seq:
                self.vocab_set.add(w)
        return self.vocab_set

    def _tally_ngrams(self, text_data):
        self.freq_uni = Counter()
        self.freq_bi = Counter()
        self.freq_tri = Counter()

        for seq in text_data:
            for w in seq:
                self.freq_uni[w] += 1

            for idx in range(len(seq) - 1):
                self.freq_bi[(seq[idx], seq[idx + 1])] += 1

            for idx in range(len(seq) - 2):
                self.freq_tri[(seq[idx], seq[idx + 1], seq[idx + 2])] += 1

        self.total_words = sum(self.freq_uni.values())

    def _calc_prob_uni(self):
        self.prob_uni = {}
        for w, count_val in self.freq_uni.items():
            self.prob_uni[w] = count_val / self.total_words

    def _calc_prob_bi(self):
        self.prob_bi = {}
        for (w1, w2), count_val in self.freq_bi.items():
            self.prob_bi[(w1, w2)] = count_val / self.freq_uni[w1]

    def _calc_prob_tri(self):
        self.prob_tri = {}
        for (w1, w2, w3), count_val in self.freq_tri.items():
            self.prob_tri[(w1, w2, w3)] = count_val / self.freq_bi[(w1, w2)]

    def train_model(self, text_data):
        self._create_vocab(text_data)
        self._tally_ngrams(text_data)
        self._calc_prob_uni()

        if self.degree >= 2:
            self._calc_prob_bi()

        if self.degree >= 3:
            self._calc_prob_tri()

        return self

    def get_prob(self, ctx, w):
        if self.degree == 1:
            return self.prob_uni.get(w, 0.0)
        elif self.degree == 2:
            if len(ctx) == 0:
                return self.prob_uni.get(w, 0.0)
            elif len(ctx) != 1:
                raise ValueError("Bigram context must contain 1 word")
            return self.prob_bi.get((ctx[0], w), 0.0)
        elif self.degree == 3:
            if len(ctx) == 0:
                return self.prob_uni.get(w, 0.0)
            elif len(ctx) == 1:
                return self.prob_bi.get((ctx[0], w), 0.0)
            elif len(ctx) != 2:
                raise ValueError("Trigram context must contain 2 words")
            return self.prob_tri.get((ctx[0], ctx[1], w), 0.0)

    def calc_seq_prob(self, seq):
        ans_prob = 1.0
        if self.degree == 1:
            for w in seq:
                ans_prob *= self.get_prob((), w)
                
        elif self.degree == 2:
            if len(seq) > 0:
                ans_prob *= self.get_prob((), seq[0])
            for idx in range(1, len(seq)):
                ctx = (seq[idx - 1],)
                ans_prob *= self.get_prob(ctx, seq[idx])
                
        elif self.degree == 3:
            if len(seq) > 0:
                ans_prob *= self.get_prob((), seq[0])
            if len(seq) >= 2:
                ans_prob *= self.get_prob((seq[0],), seq[1])
            for idx in range(2, len(seq)):
                ctx = (seq[idx - 2], seq[idx - 1])
                ans_prob *= self.get_prob(ctx, seq[idx])

        return ans_prob

    def calc_seq_log_prob(self, seq):
        log_prob_sum = 0.0
        if self.degree == 1:
            for w in seq:
                p = self.get_prob((), w)
                if p == 0: return float('-inf')
                log_prob_sum += np.log(p)
                
        elif self.degree == 2:
            if len(seq) > 0:
                p = self.get_prob((), seq[0])
                if p == 0: return float('-inf')
                log_prob_sum += np.log(p)
            for idx in range(1, len(seq)):
                ctx = (seq[idx - 1],)
                p = self.get_prob(ctx, seq[idx])
                if p == 0: return float('-inf')
                log_prob_sum += np.log(p)
                
        elif self.degree == 3:
            if len(seq) > 0:
                p = self.get_prob((), seq[0])
                if p == 0: return float('-inf')
                log_prob_sum += np.log(p)
            if len(seq) >= 2:
                p = self.get_prob((seq[0],), seq[1])
                if p == 0: return float('-inf')
                log_prob_sum += np.log(p)
            for idx in range(2, len(seq)):
                ctx = (seq[idx - 2], seq[idx - 1])
                p = self.get_prob(ctx, seq[idx])
                if p == 0: return float('-inf')
                log_prob_sum += np.log(p)

        return log_prob_sum

    def predict_next_distribution(self, ctx):
        dist_dict = {}
        for w in self.vocab_set:
            p = self.get_prob(ctx, w)
            if p > 0:
                dist_dict[w] = p

        return dict(
            sorted(
                dist_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

sample_data = [
    ["we", "use", "solar", "power"],
    ["we", "use", "wind", "energy"],
    ["they", "study", "solar", "power"]
]

lm_model = NgramTextModel(2)
lm_model.train_model(sample_data)
print(lm_model.vocab_set)
print(lm_model.freq_uni)