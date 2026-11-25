import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import string
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizerFast

def normalize_answer(text):
    if not text:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def compute_exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))

def compute_token_f1(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = sum(min(count, gold_counts.get(t, 0)) for t, count in common.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * (precision * recall) / (precision + recall)

class BertForSpanQA(nn.Module):
    def __init__(self, config=None, pretrained_name="bert-base-uncased"):
        super().__init__()
        if config is None:
            config = BertConfig.from_pretrained(pretrained_name, local_files_only=True)
        self.config = config
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

def compute_qa_loss(start_logits, end_logits, start_positions, end_positions):
    ignored_index = start_logits.size(1)
    start_positions = start_positions.clamp(0, ignored_index)
    end_positions = end_positions.clamp(0, ignored_index)
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    return (start_loss + end_loss) / 2.0

def extract_answer_span(start_logits, end_logits, input_ids, tokenizer, max_answer_length=30):
    start_probs = torch.softmax(start_logits.detach(), dim=-1)
    end_probs = torch.softmax(end_logits.detach(), dim=-1)
    best_score = -1.0
    best_span = (0, 0)
    seq_len = start_logits.size(0)
    for start_idx in range(seq_len):
        for end_idx in range(start_idx, min(seq_len, start_idx + max_answer_length)):
            score = float(start_probs[start_idx] * end_probs[end_idx])
            if score > best_score:
                best_score = score
                best_span = (start_idx, end_idx)
    s_idx, e_idx = best_span
    span_token_ids = input_ids[s_idx:e_idx + 1].tolist()
    answer_text = tokenizer.decode(span_token_ids, skip_special_tokens=True).strip()
    return answer_text, s_idx, e_idx, best_score

if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", local_files_only=True)
    config = BertConfig.from_pretrained("bert-base-uncased", local_files_only=True)
    config.num_hidden_layers = 2
    model = BertForSpanQA(config=config)
    question = "What is the capital of France?"
    context = "The capital of France is Paris."
    encoded = tokenizer(
        question,
        context,
        max_length=64,
        padding="max_length",
        truncation="only_second",
        return_tensors="pt"
    )
    s_logits, e_logits = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        token_type_ids=encoded.get("token_type_ids")
    )
    start_pos = torch.tensor([8], dtype=torch.long)
    end_pos = torch.tensor([8], dtype=torch.long)
    loss = compute_qa_loss(s_logits, e_logits, start_pos, end_pos)
    pred_ans, s_idx, e_idx, score = extract_answer_span(
        s_logits[0], e_logits[0], encoded["input_ids"][0], tokenizer
    )
    em = compute_exact_match("Paris", "paris")
    f1 = compute_token_f1("Paris France", "France")
    print(f"Logits shape: {s_logits.shape}, Loss: {loss.item():.4f}")
    print(f"Extracted span indices: ({s_idx}, {e_idx}), Score: {score:.4f}")
    print(f"Test EM: {em:.1f}, Test F1: {f1:.4f}")
