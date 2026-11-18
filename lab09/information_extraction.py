import json
import re
from collections import defaultdict
from baseline_ner import BaselineNER, compute_prf, extract_entities, get_sample_conll_data

TYPE_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION"
}

def tokenize_text(text):
    return re.findall(r"[A-Za-z0-9'-]+|[.,!?;:]", text)

def bio_to_structured(tokens, tags):
    result = {"PERSON": [], "ORG": [], "LOCATION": []}
    for ent_type, start, end in extract_entities(tags):
        mapped = TYPE_MAP.get(ent_type)
        if mapped is not None:
            mention = " ".join(tokens[start:end + 1])
            if mention not in result[mapped]:
                result[mapped].append(mention)
    return result

class InformationExtractor:
    def __init__(self, ner_model):
        self.ner_model = ner_model

    def extract(self, text):
        tokens = tokenize_text(text)
        tags = self.ner_model.predict([tokens])[0]
        structured = bio_to_structured(tokens, tags)
        return {
            "text": text,
            "tokens": tokens,
            "tags": tags,
            "entities": structured
        }

def get_news_ie_benchmark_30():
    return [
        ("Barack Obama visited Paris.", {"PERSON": ["Barack Obama"], "ORG": [], "LOCATION": ["Paris"]}),
        ("Elon Musk founded SpaceX in California.", {"PERSON": ["Elon Musk"], "ORG": ["SpaceX"], "LOCATION": ["California"]}),
        ("Google acquired DeepMind in London.", {"PERSON": [], "ORG": ["Google", "DeepMind"], "LOCATION": ["London"]}),
        ("Tim Cook announced Apple products in Cupertino.", {"PERSON": ["Tim Cook"], "ORG": ["Apple"], "LOCATION": ["Cupertino"]}),
        ("Microsoft opened offices in Berlin.", {"PERSON": [], "ORG": ["Microsoft"], "LOCATION": ["Berlin"]}),
        ("Angela Merkel spoke at United Nations in New York.", {"PERSON": ["Angela Merkel"], "ORG": ["United Nations"], "LOCATION": ["New York"]}),
        ("Sundar Pichai leads Google from Mountain View.", {"PERSON": ["Sundar Pichai"], "ORG": ["Google"], "LOCATION": ["Mountain View"]}),
        ("Satya Nadella visited Tokyo for Microsoft.", {"PERSON": ["Satya Nadella"], "ORG": ["Microsoft"], "LOCATION": ["Tokyo"]}),
        ("Jeff Bezos founded Amazon in Seattle.", {"PERSON": ["Jeff Bezos"], "ORG": ["Amazon"], "LOCATION": ["Seattle"]}),
        ("Mark Zuckerberg spoke in Washington.", {"PERSON": ["Mark Zuckerberg"], "ORG": [], "LOCATION": ["Washington"]}),
        ("Reuters reported inflation in Frankfurt.", {"PERSON": [], "ORG": ["Reuters"], "LOCATION": ["Frankfurt"]}),
        ("BBC interviewed scientists in Oxford.", {"PERSON": [], "ORG": ["BBC"], "LOCATION": ["Oxford"]}),
        ("OpenAI collaborated with Microsoft in San Francisco.", {"PERSON": [], "ORG": ["OpenAI", "Microsoft"], "LOCATION": ["San Francisco"]}),
        ("Emmanuel Macron addressed voters in Paris.", {"PERSON": ["Emmanuel Macron"], "ORG": [], "LOCATION": ["Paris"]}),
        ("Toyota built cars in Osaka.", {"PERSON": [], "ORG": ["Toyota"], "LOCATION": ["Osaka"]}),
        ("World Health Organization met in Geneva.", {"PERSON": [], "ORG": ["World Health Organization"], "LOCATION": ["Geneva"]}),
        ("NATO officials gathered in Brussels.", {"PERSON": [], "ORG": ["NATO"], "LOCATION": ["Brussels"]}),
        ("Amazon expanded operations in London.", {"PERSON": [], "ORG": ["Amazon"], "LOCATION": ["London"]}),
        ("University of California Berkeley hosted engineers.", {"PERSON": [], "ORG": ["University of California Berkeley"], "LOCATION": []}),
        ("United Nations Security Council convened in New York.", {"PERSON": [], "ORG": ["United Nations Security Council"], "LOCATION": ["New York"]}),
        ("Elon Musk visited Berlin for SpaceX.", {"PERSON": ["Elon Musk"], "ORG": ["SpaceX"], "LOCATION": ["Berlin"]}),
        ("Tim Cook met leaders in Brussels.", {"PERSON": ["Tim Cook"], "ORG": [], "LOCATION": ["Brussels"]}),
        ("Google opened a lab in Tokyo.", {"PERSON": [], "ORG": ["Google"], "LOCATION": ["Tokyo"]}),
        ("Satya Nadella spoke at Oxford.", {"PERSON": ["Satya Nadella"], "ORG": [], "LOCATION": ["Oxford"]}),
        ("Reuters interviewed Jeff Bezos in Seattle.", {"PERSON": ["Jeff Bezos"], "ORG": ["Reuters"], "LOCATION": ["Seattle"]}),
        ("Jensen Huang unveiled Nvidia chips in Taipei.", {"PERSON": ["Jensen Huang"], "ORG": ["Nvidia"], "LOCATION": ["Taipei"]}),
        ("Christine Lagarde addressed European Central Bank in Frankfurt.", {"PERSON": ["Christine Lagarde"], "ORG": ["European Central Bank"], "LOCATION": ["Frankfurt"]}),
        ("Sam Altman visited London for OpenAI.", {"PERSON": ["Sam Altman"], "ORG": ["OpenAI"], "LOCATION": ["London"]}),
        ("Sony expanded studios in Tokyo.", {"PERSON": [], "ORG": ["Sony"], "LOCATION": ["Tokyo"]}),
        ("International Monetary Fund met officials in Washington.", {"PERSON": [], "ORG": ["International Monetary Fund"], "LOCATION": ["Washington"]})
    ]

def evaluate_ie_benchmark(extractor, benchmark):
    total_tp = total_fp = total_fn = 0
    per_cat = defaultdict(lambda: [0, 0, 0])

    for text, gold_dict in benchmark:
        pred_dict = extractor.extract(text)["entities"]
        for cat in ["PERSON", "ORG", "LOCATION"]:
            gold_set = set(gold_dict.get(cat, []))
            pred_set = set(pred_dict.get(cat, []))
            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            per_cat[cat][0] += tp
            per_cat[cat][1] += fp
            per_cat[cat][2] += fn
            total_tp += tp
            total_fp += fp
            total_fn += fn

    overall_p, overall_r, overall_f1 = compute_prf(total_tp, total_fp, total_fn)
    category_scores = {}
    for cat in ["PERSON", "ORG", "LOCATION"]:
        tp, fp, fn = per_cat[cat]
        p, r, f1 = compute_prf(tp, fp, fn)
        category_scores[cat] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    return {
        "num_sentences": len(benchmark),
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "by_category": category_scores
    }

if __name__ == "__main__":
    train_data, _ = get_sample_conll_data()
    ner = BaselineNER()
    ner.fit([x for x, _ in train_data], [y for _, y in train_data])
    extractor = InformationExtractor(ner)

    demo_out = extractor.extract("Barack Obama visited Paris.")
    print("Structured IE Output Example:")
    print(json.dumps(demo_out["entities"], indent=2))

    benchmark_30 = get_news_ie_benchmark_30()
    eval_res = evaluate_ie_benchmark(extractor, benchmark_30)
    print(f"\nMini IE Evaluation on {eval_res['num_sentences']} News Sentences (Entity-Level):")
    print(f"Overall -> Precision: {eval_res['precision']:.4f} | Recall: {eval_res['recall']:.4f} | F1: {eval_res['f1']:.4f} (TP={eval_res['tp']}, FP={eval_res['fp']}, FN={eval_res['fn']})")
    for cat, scores in eval_res["by_category"].items():
        print(f"  {cat:<8} -> P: {scores['precision']:.4f} | R: {scores['recall']:.4f} | F1: {scores['f1']:.4f}")
