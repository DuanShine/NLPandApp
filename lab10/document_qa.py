import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import torch
from transformers import BertConfig, BertTokenizerFast
from qa_model import BertForSpanQA, compute_exact_match, compute_token_f1, normalize_answer

class DocumentQASystem:
    def __init__(self, model=None, tokenizer=None, device="cpu"):
        self.device = torch.device(device)
        if tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)
        else:
            self.tokenizer = tokenizer
        if model is None:
            config = BertConfig.from_pretrained("bert-base-uncased", local_files_only=True)
            config.num_hidden_layers = 4
            self.model = BertForSpanQA(config=config)
        else:
            self.model = model
        self.model.to(self.device)
        self.model.eval()

    def predict(self, question, context, max_seq_length=384, doc_stride=128, max_answer_length=30):
        encodings = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        offset_mapping = encodings.pop("offset_mapping")
        encodings.pop("overflow_to_sample_mapping", None)
        num_spans = encodings["input_ids"].size(0)
        best_score = -1.0
        best_answer = ""
        best_char_start = -1
        best_char_end = -1
        best_window = 0
        with torch.no_grad():
            for i in range(num_spans):
                input_ids = encodings["input_ids"][i:i + 1].to(self.device)
                attention_mask = encodings["attention_mask"][i:i + 1].to(self.device)
                token_type_ids = encodings.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids[i:i + 1].to(self.device)
                s_logits, e_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                s_logits = s_logits[0].cpu()
                e_logits = e_logits[0].cpu()
                offsets = offset_mapping[i].tolist()
                seq_ids = encodings.sequence_ids(i)
                cls_score = float(torch.softmax(s_logits, dim=-1)[0] * torch.softmax(e_logits, dim=-1)[0])
                s_probs = torch.softmax(s_logits, dim=-1)
                e_probs = torch.softmax(e_logits, dim=-1)
                for s_idx in range(len(offsets)):
                    if seq_ids[s_idx] != 1:
                        continue
                    if offsets[s_idx] == [0, 0]:
                        continue
                    for e_idx in range(s_idx, min(len(offsets), s_idx + max_answer_length)):
                        if seq_ids[e_idx] != 1:
                            continue
                        if offsets[e_idx] == [0, 0]:
                            continue
                        score = float(s_probs[s_idx] * e_probs[e_idx])
                        if score > best_score:
                            best_score = score
                            best_char_start = offsets[s_idx][0]
                            best_char_end = offsets[e_idx][1]
                            best_answer = context[best_char_start:best_char_end].strip()
                            best_window = i
        return {
            "question": question,
            "answer": best_answer,
            "char_start": best_char_start,
            "char_end": best_char_end,
            "confidence": best_score,
            "window_index": best_window
        }

    def evaluate_benchmark(self, dataset):
        results = []
        total_em = 0.0
        total_f1 = 0.0
        for sample in dataset:
            question = sample["question"]
            context = sample["context"]
            gold = sample["answer"]
            pred_dict = self.predict(question, context)
            pred_text = pred_dict["answer"]
            em = compute_exact_match(pred_text, gold)
            f1 = compute_token_f1(pred_text, gold)
            total_em += em
            total_f1 += f1
            results.append({
                "question": question,
                "gold": gold,
                "prediction": pred_text,
                "em": em,
                "f1": f1,
                "confidence": pred_dict["confidence"]
            })
        n = max(1, len(dataset))
        return {
            "exact_match": total_em / n * 100.0,
            "f1": total_f1 / n * 100.0,
            "num_samples": n,
            "details": results
        }

def get_demo_qa_samples():
    return [
        {
            "context": "The capital of France is Paris. Situated on the Seine River, Paris is the center of French politics and commerce.",
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "context": "Alan Turing was an English mathematician and computer scientist. He developed the concept of the Turing machine in 1936.",
            "question": "When did Turing develop the Turing machine concept?",
            "answer": "1936"
        },
        {
            "context": "The Apollo 11 mission landed astronauts Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969.",
            "question": "Who were the astronauts that landed on the Moon?",
            "answer": "Neil Armstrong and Buzz Aldrin"
        },
        {
            "context": "Photosynthesis is the biological process used by plants and other organisms to convert light energy into chemical energy.",
            "question": "What process converts light energy into chemical energy?",
            "answer": "Photosynthesis"
        },
        {
            "context": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. Its elevation is 8,848.86 meters.",
            "question": "What is the elevation of Mount Everest?",
            "answer": "8,848.86 meters"
        }
    ]

if __name__ == "__main__":
    qa_sys = DocumentQASystem()
    samples = get_demo_qa_samples()
    print("Running Document QA System Evaluation on Benchmark...")
    eval_res = qa_sys.evaluate_benchmark(samples)
    print(f"Evaluated on {eval_res['num_samples']} samples:")
    for d in eval_res["details"]:
        print(f"Q: {d['question']}")
        print(f"  Gold: {d['gold']} | Pred: {d['prediction']} | Conf: {d['confidence']:.4f}")
        print(f"  EM: {d['em']:.0f} | F1: {d['f1']:.4f}")
    print(f"Overall EM: {eval_res['exact_match']:.2f}% | Overall F1: {eval_res['f1']:.2f}%")
