from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]

def extract_token_features(sentence, idx):
    word = sentence[idx]
    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.prefix2": word[:2].lower(),
        "word.suffix2": word[-2:].lower(),
        "word.suffix3": word[-3:].lower(),
        "word.has_hyphen": "-" in word,
        "word.is_punct": word in [".", ",", "!", "?", ";", ":"]
    }
    if idx > 0:
        prev_word = sentence[idx - 1]
        features["prev.lower"] = prev_word.lower()
        features["prev.istitle"] = prev_word.istitle()
        features["prev.isupper"] = prev_word.isupper()
    else:
        features["BOS"] = True

    if idx < len(sentence) - 1:
        next_word = sentence[idx + 1]
        features["next.lower"] = next_word.lower()
        features["next.istitle"] = next_word.istitle()
        features["next.isupper"] = next_word.isupper()
    else:
        features["EOS"] = True

    return features

def extract_entities(tags):
    entities = []
    current_type = None
    start_idx = None

    for idx, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None:
                entities.append((current_type, start_idx, idx - 1))
            current_type = tag[2:]
            start_idx = idx
        elif tag.startswith("I-"):
            ent_type = tag[2:]
            if current_type == ent_type:
                continue
            if current_type is not None:
                entities.append((current_type, start_idx, idx - 1))
            current_type = ent_type
            start_idx = idx
        else:
            if current_type is not None:
                entities.append((current_type, start_idx, idx - 1))
                current_type = None
                start_idx = None

    if current_type is not None:
        entities.append((current_type, start_idx, len(tags) - 1))

    return entities

def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def compute_entity_metrics(gold_seqs, pred_seqs):
    total_tp = total_fp = total_fn = 0
    by_type = defaultdict(lambda: [0, 0, 0])
    by_len = defaultdict(lambda: [0, 0, 0])

    for gold_tags, pred_tags in zip(gold_seqs, pred_seqs):
        gold_set = set(extract_entities(gold_tags))
        pred_set = set(extract_entities(pred_tags))

        tp_set = gold_set & pred_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set

        total_tp += len(tp_set)
        total_fp += len(fp_set)
        total_fn += len(fn_set)

        for ent_type, start, end in tp_set:
            by_type[ent_type][0] += 1
            length_key = str(end - start + 1) if (end - start + 1) < 4 else "4+"
            by_len[length_key][0] += 1

        for ent_type, start, end in fp_set:
            by_type[ent_type][1] += 1
            length_key = str(end - start + 1) if (end - start + 1) < 4 else "4+"
            by_len[length_key][1] += 1

        for ent_type, start, end in fn_set:
            by_type[ent_type][2] += 1
            length_key = str(end - start + 1) if (end - start + 1) < 4 else "4+"
            by_len[length_key][2] += 1

    overall_p, overall_r, overall_f1 = compute_prf(total_tp, total_fp, total_fn)

    type_metrics = {}
    for ent_type in ["PER", "ORG", "LOC", "MISC"]:
        tp, fp, fn = by_type[ent_type]
        p, r, f1 = compute_prf(tp, fp, fn)
        type_metrics[ent_type] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    length_metrics = {}
    for length_key in ["1", "2", "3", "4+"]:
        tp, fp, fn = by_len[length_key]
        p, r, f1 = compute_prf(tp, fp, fn)
        length_metrics[length_key] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    flat_gold = [tag for seq in gold_seqs for tag in seq]
    flat_pred = [tag for seq in pred_seqs for tag in seq]
    token_acc = accuracy_score(flat_gold, flat_pred)

    return {
        "token_accuracy": token_acc,
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "by_type": type_metrics,
        "by_length": length_metrics
    }

class BaselineNER:
    def __init__(self, c_value=1.0, max_iter=500):
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = LogisticRegression(C=c_value, max_iter=max_iter, random_state=42)

    def fit(self, sentences, tag_seqs):
        X_dicts = []
        y_labels = []
        for sent, tags in zip(sentences, tag_seqs):
            for idx in range(len(sent)):
                X_dicts.append(extract_token_features(sent, idx))
                y_labels.append(tags[idx])
        X_vec = self.vectorizer.fit_transform(X_dicts)
        self.classifier.fit(X_vec, y_labels)
        return self

    def predict(self, sentences):
        predictions = []
        for sent in sentences:
            X_dicts = [extract_token_features(sent, idx) for idx in range(len(sent))]
            X_vec = self.vectorizer.transform(X_dicts)
            preds = self.classifier.predict(X_vec).tolist()
            predictions.append(preds)
        return predictions

def get_sample_conll_data():
    train_data = [
        (["Barack", "Obama", "visited", "Paris", "."], ["B-PER", "I-PER", "O", "B-LOC", "O"]),
        (["Elon", "Musk", "founded", "SpaceX", "in", "California", "."], ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "O"]),
        (["Google", "acquired", "DeepMind", "in", "London", "."], ["B-ORG", "O", "B-ORG", "O", "B-LOC", "O"]),
        (["Angela", "Merkel", "spoke", "at", "United", "Nations", "in", "New", "York", "."], ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "O", "B-LOC", "I-LOC", "O"]),
        (["Microsoft", "opened", "a", "lab", "in", "Berlin", "."], ["B-ORG", "O", "O", "O", "O", "B-LOC", "O"]),
        (["Tim", "Cook", "announced", "Apple", "revenue", "in", "Cupertino", "."], ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "B-LOC", "O"]),
        (["Reuters", "reported", "German", "inflation", "in", "Frankfurt", "."], ["B-ORG", "O", "B-MISC", "O", "O", "B-LOC", "O"]),
        (["University", "of", "California", "Berkeley", "hosted", "European", "researchers", "."], ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "B-MISC", "O", "O"]),
        (["Sundar", "Pichai", "leads", "Google", "from", "Mountain", "View", "."], ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "I-LOC", "O"]),
        (["French", "officials", "met", "in", "Brussels", "with", "NATO", "."], ["B-MISC", "O", "O", "O", "B-LOC", "O", "B-ORG", "O"]),
        (["Satya", "Nadella", "visited", "Tokyo", "for", "Microsoft", "."], ["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "O"]),
        (["BBC", "interviewed", "British", "scientists", "in", "Oxford", "."], ["B-ORG", "O", "B-MISC", "O", "O", "B-LOC", "O"]),
        (["Amazon", "expanded", "offices", "in", "Seattle", "."], ["B-ORG", "O", "O", "O", "B-LOC", "O"]),
        (["Jeff", "Bezos", "founded", "Amazon", "in", "Seattle", "."], ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "O"]),
        (["World", "Health", "Organization", "warned", "Asian", "countries", "."], ["B-ORG", "I-ORG", "I-ORG", "O", "B-MISC", "O", "O"]),
        (["United", "Nations", "Security", "Council", "met", "in", "Geneva", "."], ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "O"]),
        (["Mark", "Zuckerberg", "spoke", "in", "Washington", "."], ["B-PER", "I-PER", "O", "O", "B-LOC", "O"]),
        (["Toyota", "built", "Japanese", "cars", "in", "Osaka", "."], ["B-ORG", "O", "B-MISC", "O", "O", "B-LOC", "O"]),
        (["OpenAI", "collaborated", "with", "Microsoft", "in", "San", "Francisco", "."], ["B-ORG", "O", "O", "B-ORG", "O", "B-LOC", "I-LOC", "O"]),
        (["Emmanuel", "Macron", "addressed", "French", "voters", "in", "Paris", "."], ["B-PER", "I-PER", "O", "B-MISC", "O", "O", "B-LOC", "O"])
    ]
    test_data = [
        (["Barack", "Obama", "spoke", "in", "Berlin", "."], ["B-PER", "I-PER", "O", "O", "B-LOC", "O"]),
        (["Google", "opened", "offices", "in", "New", "York", "."], ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC", "O"]),
        (["Tim", "Cook", "visited", "Tokyo", "for", "Apple", "."], ["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "O"]),
        (["United", "Nations", "met", "European", "leaders", "in", "Geneva", "."], ["B-ORG", "I-ORG", "O", "B-MISC", "O", "O", "B-LOC", "O"]),
        (["Elon", "Musk", "leads", "SpaceX", "from", "California", "."], ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "O"]),
        (["Reuters", "reported", "British", "trade", "in", "London", "."], ["B-ORG", "O", "B-MISC", "O", "O", "B-LOC", "O"]),
        (["Jensen", "Huang", "launched", "Nvidia", "chips", "in", "Taipei", "."], ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "B-LOC", "O"]),
        (["International", "Monetary", "Fund", "advised", "Spanish", "banks", "in", "Madrid", "."], ["B-ORG", "I-ORG", "I-ORG", "O", "B-MISC", "O", "O", "B-LOC", "O"])
    ]
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = get_sample_conll_data()
    X_train = [x for x, _ in train_data]
    y_train = [y for _, y in train_data]
    X_test = [x for x, _ in test_data]
    y_test = [y for _, y in test_data]

    model = BaselineNER()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_entity_metrics(y_test, preds)

    print(f"Token Accuracy  : {metrics['token_accuracy']:.4f}")
    print(f"Entity Precision: {metrics['precision']:.4f}")
    print(f"Entity Recall   : {metrics['recall']:.4f}")
    print(f"Entity F1       : {metrics['f1']:.4f}")
    print("Entity-Level Check ('New York' Boundary Test):")
    demo = compute_entity_metrics([["B-LOC", "I-LOC"]], [["B-LOC", "O"]])
    print(f"  Token Acc = {demo['token_accuracy']:.2f} | Entity F1 = {demo['f1']:.2f} (TP={demo['tp']}, FP={demo['fp']}, FN={demo['fn']})")
