"""LAB 08 - Pretrained Transformers and BERT Fine-Tuning.

Library + CLI used by ``fine_tuning.ipynb``.

Pipeline (section 13 of the lab):
    text -> tokenizer -> input_ids / attention_mask -> pretrained Transformer
         -> contextual representations -> [CLS] vector -> Linear -> logits -> softmax

Experiments implemented here:
    1. Pretrained (fine-tuned) vs Transformer trained from scratch
    2. Feature extraction (frozen encoder) vs full fine-tuning
    3. Amount of training data (10/25/50/100 %)
    4. Maximum sequence length (64/128/256)
    5. Domain shift (train on IMDb movie reviews, test on tweets)

Run everything from the command line:
    python pretrained_classification.py            # all experiments
    python pretrained_classification.py --quick    # tiny smoke test
"""
from __future__ import annotations

import argparse
import os
import platform
import random
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["negative", "positive"]


# --------------------------------------------------------------------------- #
# Configuration / reproducibility
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    seed: int = 42
    n_train: int = 4000          # IMDb train subset (100 %)
    n_val: int = 1000            # IMDb validation subset (from the train split)
    n_test: int = 2000           # IMDb test subset
    n_ood: int = 1000            # out-of-domain test (tweets)
    max_len: int = 128
    batch_size: int = 32
    # full fine-tuning
    ft_epochs: int = 2
    ft_lr: float = 2e-5
    # frozen encoder (feature extraction) - only the linear head is trained
    frozen_epochs: int = 30
    frozen_lr: float = 1e-3
    # small Transformer trained from scratch
    scratch_epochs: int = 5
    scratch_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def hardware_info() -> str:
    cpu = platform.processor() or platform.machine()
    ram = psutil.virtual_memory().total / 2**30
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU"
    return f"{cpu}, {os.cpu_count()} logical cores, {ram:.0f} GB RAM, {gpu}"


def rss_mb() -> float:
    """Peak GPU memory (CUDA) or current process RSS (CPU), in MB."""
    if DEVICE.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    return psutil.Process().memory_info().rss / 2**20


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_imdb(cfg: Config) -> dict:
    """Fixed, seeded IMDb subsets. Train subsets for 10/25/50 % are prefixes of
    the shuffled train set, so they are nested and validation/test stay fixed."""
    ds = load_dataset("stanfordnlp/imdb")
    train_full = ds["train"].shuffle(seed=cfg.seed)
    train = train_full.select(range(cfg.n_train))
    val = train_full.select(range(cfg.n_train, cfg.n_train + cfg.n_val))
    test = ds["test"].shuffle(seed=cfg.seed).select(range(cfg.n_test))
    return {
        "train": (list(train["text"]), list(train["label"])),
        "val": (list(val["text"]), list(val["label"])),
        "test": (list(test["text"]), list(test["label"])),
    }


def load_tweets(cfg: Config) -> tuple[list[str], list[int]]:
    """Out-of-domain set: TweetEval sentiment, neutral removed, 0=neg, 1=pos."""
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
    ds = ds.filter(lambda e: e["label"] != 1).shuffle(seed=cfg.seed)
    ds = ds.select(range(min(cfg.n_ood, len(ds))))
    return list(ds["text"]), [1 if lab == 2 else 0 for lab in ds["label"]]


def take_fraction(split: tuple[list, list], frac: float) -> tuple[list, list]:
    k = int(round(len(split[0]) * frac))
    return split[0][:k], split[1][:k]


def encode(tokenizer, texts, labels, max_len: int) -> TensorDataset:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class PretrainedClassifier(nn.Module):
    """Pretrained encoder + task-specific linear head on the [CLS] vector."""

    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_labels)

    def cls_vector(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]  # [CLS] is always the first token

    def forward(self, input_ids, attention_mask):
        return self.head(self.dropout(self.cls_vector(input_ids, attention_mask)))


class ScratchTransformerClassifier(nn.Module):
    """Small Transformer encoder with random initialisation (LAB 07 style).
    Uses the same tokenizer/vocabulary as the pretrained model so that the
    only difference is the initial weights and the model size."""

    def __init__(self, vocab_size, max_len, pad_id=0, d_model=128, n_heads=4,
                 n_layers=2, d_ff=256, num_labels=2, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos)[None]
        x = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        # masked mean pooling: a randomly initialised [CLS] has no pretrained meaning
        m = attention_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(self.norm(pooled))


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


@torch.no_grad()
def predict_proba(model, ds: TensorDataset, batch_size: int = 64) -> np.ndarray:
    model.eval()
    probs = []
    for ids, mask, _ in DataLoader(ds, batch_size=batch_size):
        logits = model(ids.to(DEVICE), mask.to(DEVICE))
        probs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(probs).numpy()


def train_model(model, train_ds, val_ds, cfg: Config, epochs: int, lr: float, log=print) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    total_steps = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss()
    peak = rss_mb()
    history = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(ids, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            n += len(y)
            peak = max(peak, rss_mb())
        val_acc = accuracy_score(val_ds.tensors[2].numpy(), predict_proba(model, val_ds).argmax(1))
        history.append({"epoch": ep, "train_loss": running / n, "val_acc": val_acc})
        log(f"  epoch {ep}/{epochs}  loss={running / n:.4f}  val_acc={val_acc:.4f}  "
            f"({time.perf_counter() - t0:.0f}s)")
    return {"train_time_s": time.perf_counter() - t0, "peak_rss_mb": peak, "history": history}


@torch.no_grad()
def extract_cls(model: PretrainedClassifier, ds: TensorDataset, batch_size: int = 64):
    model.eval()
    feats = []
    for ids, mask, _ in DataLoader(ds, batch_size=batch_size):
        feats.append(model.cls_vector(ids.to(DEVICE), mask.to(DEVICE)))
    return torch.cat(feats)


def train_frozen(model: PretrainedClassifier, train_ds, val_ds, cfg: Config, log=print) -> dict:
    """Feature extraction: the encoder is frozen, so its [CLS] vectors never
    change. We compute them once and train only the linear head on top."""
    for p in model.encoder.parameters():
        p.requires_grad = False
    peak = rss_mb()
    t0 = time.perf_counter()
    x_tr, y_tr = extract_cls(model, train_ds), train_ds.tensors[2].to(DEVICE)
    x_va, y_va = extract_cls(model, val_ds), val_ds.tensors[2].to(DEVICE)
    peak = max(peak, rss_mb())
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW(head_params, lr=cfg.frozen_lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(cfg.seed)
    history = []
    for ep in range(1, cfg.frozen_epochs + 1):
        model.head.train()
        model.dropout.train()
        perm = torch.randperm(len(x_tr), generator=g).to(DEVICE)
        running = 0.0
        for i in range(0, len(x_tr), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model.head(model.dropout(x_tr[idx])), y_tr[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_acc = (model.head(x_va).argmax(1) == y_va).float().mean().item()
        history.append({"epoch": ep, "train_loss": running / len(x_tr), "val_acc": val_acc})
    log(f"  frozen head: final loss={history[-1]['train_loss']:.4f} val_acc={history[-1]['val_acc']:.4f}")
    return {"train_time_s": time.perf_counter() - t0, "peak_rss_mb": peak, "history": history}


@dataclass
class RunResult:
    row: dict
    model: nn.Module
    probs: np.ndarray       # test probabilities (IMDb test)
    history: list


def run(
    kind: str,                 # "pretrained" | "scratch"
    mode: str,                 # "fine-tuning" | "frozen" | "from scratch"
    data: dict,
    tokenizer,
    cfg: Config,
    experiment: str,
    train_frac: float = 1.0,
    max_len: int | None = None,
    log=print,
) -> RunResult:
    set_seed(cfg.seed)
    max_len = max_len or cfg.max_len
    tr_texts, tr_labels = take_fraction(data["train"], train_frac)
    train_ds = encode(tokenizer, tr_texts, tr_labels, max_len)
    val_ds = encode(tokenizer, *data["val"], max_len)
    test_ds = encode(tokenizer, *data["test"], max_len)
    log(f"[{experiment}] {kind}/{mode}  train={len(tr_texts)}  max_len={max_len}")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if kind == "pretrained":
        model = PretrainedClassifier(cfg.model_name).to(DEVICE)
        if mode == "frozen":
            info = train_frozen(model, train_ds, val_ds, cfg, log)
            epochs, lr = cfg.frozen_epochs, cfg.frozen_lr
        else:
            epochs, lr = cfg.ft_epochs, cfg.ft_lr
            info = train_model(model, train_ds, val_ds, cfg, epochs, lr, log)
    else:
        model = ScratchTransformerClassifier(
            tokenizer.vocab_size, max_len, pad_id=tokenizer.pad_token_id
        ).to(DEVICE)
        epochs, lr = cfg.scratch_epochs, cfg.scratch_lr
        info = train_model(model, train_ds, val_ds, cfg, epochs, lr, log)

    t0 = time.perf_counter()
    probs = predict_proba(model, test_ds)
    infer_s = time.perf_counter() - t0
    y_true = data["test"][1]
    m = compute_metrics(y_true, probs.argmax(1))
    total, trainable = count_params(model)
    row = {
        "experiment": experiment,
        "model": "DistilBERT (pretrained)" if kind == "pretrained" else "Transformer from scratch",
        "mode": mode,
        "encoder_trainable": mode != "frozen",
        "train_frac": train_frac,
        "n_train": len(tr_texts),
        "max_len": max_len,
        "epochs": epochs,
        "lr": lr,
        "eval_set": "IMDb test",
        **m,
        "val_acc": info["history"][-1]["val_acc"],
        "train_time_s": round(info["train_time_s"], 1),
        "test_infer_s": round(infer_s, 1),
        "peak_mem_mb": round(info["peak_rss_mb"]),
        "device": DEVICE.type,
        "params_total": total,
        "params_trainable": trainable,
    }
    log(f"  -> acc={m['accuracy']:.4f} f1={m['f1']:.4f} time={row['train_time_s']}s")
    return RunResult(row, model, probs, info["history"])


def evaluate_on(model, tokenizer, texts, labels, max_len) -> tuple[dict, np.ndarray]:
    probs = predict_proba(model, encode(tokenizer, texts, labels, max_len))
    return compute_metrics(labels, probs.argmax(1)), probs


# --------------------------------------------------------------------------- #
# Plots and analysis helpers
# --------------------------------------------------------------------------- #
def plot_confusion(y_true, y_pred, path, title="Confusion matrix (IMDb test)"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig, cm


def plot_data_efficiency(df: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for metric, ax in zip(["accuracy", "f1"], axes):
        for model, g in df.groupby("model"):
            g = g.sort_values("train_frac")
            ax.plot(g["train_frac"] * 100, g[metric], marker="o", label=model)
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
        ax.set_xlabel("Fraction of training data (log scale)")
        ax.set_ylabel(metric)
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
        ax.set_title(f"Data efficiency - {metric}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


CONF_BINS = [(0.5, 0.7, "Low (0.5-0.7)"), (0.7, 0.9, "Medium (0.7-0.9)"), (0.9, 1.01, "High (>=0.9)")]


def confidence_table(y_true, probs) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    rows = []
    for lo, hi, name in CONF_BINS:
        sel = (conf >= lo) & (conf < hi)
        rows.append({
            "bin": name,
            "n": int(sel.sum()),
            "share": sel.mean(),
            "accuracy": correct[sel].mean() if sel.any() else np.nan,
            "n_wrong": int((~correct & sel).sum()),
        })
    return pd.DataFrame(rows)


def plot_confidence(y_true, probs, path):
    y_true = np.asarray(y_true)
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    table = confidence_table(y_true, probs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0.5, 1.0, 21)
    axes[0].hist(conf[correct], bins=bins, alpha=0.7, label="correct")
    axes[0].hist(conf[~correct], bins=bins, alpha=0.7, label="wrong")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("confidence = max softmax probability")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("Confidence of correct vs wrong predictions")
    axes[0].legend()
    bars = axes[1].bar(table["bin"], table["accuracy"], color=["#d62728", "#ff7f0e", "#2ca02c"])
    for b, (_, r) in zip(bars, table.iterrows()):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"acc={r['accuracy']:.2f}\nn={r['n']}, wrong={r['n_wrong']}",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("accuracy in bin")
    axes[1].set_title("Accuracy per confidence bin")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig, table


def pick_errors(texts, y_true, probs, n_fp=3, n_fn=3, n_conf=2) -> pd.DataFrame:
    """Pick the most confident wrong predictions first, then the next FP / FN
    (also ranked by confidence) so every example is different."""
    y_true = np.asarray(y_true)
    pred = probs.argmax(1)
    conf = probs.max(1)
    wrong = np.where(pred != y_true)[0]
    wrong = wrong[np.argsort(-conf[wrong])]
    chosen, kinds = list(wrong[:n_conf]), ["very confident & wrong"] * min(n_conf, len(wrong))
    rest = [i for i in wrong if i not in chosen]
    fps = [i for i in rest if pred[i] == 1][:n_fp]
    fns = [i for i in rest if pred[i] == 0][:n_fn]
    chosen += fps + fns
    kinds += ["false positive"] * len(fps) + ["false negative"] * len(fns)
    return pd.DataFrame({
        "idx": chosen,
        "selection": kinds,
        "text": [texts[i] for i in chosen],
        "true": [LABELS[y_true[i]] for i in chosen],
        "pred": [LABELS[pred[i]] for i in chosen],
        "confidence": [round(float(conf[i]), 4) for i in chosen],
    })


def write_error_table(df: pd.DataFrame, path, max_chars=600):
    def clean(t):
        t = t.replace("<br />", " ").replace("|", "\\|").replace("\n", " ")
        return t if len(t) <= max_chars else t[:max_chars] + " ..."

    lines = [
        "| # | Selection | Text | True | Pred | Confidence | Error type | Hypothesis |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, r in enumerate(df.itertuples(), 1):
        lines.append(f"| {k} | {r.selection} | {clean(r.text)} | {r.true} | {r.pred} | "
                     f"{r.confidence:.3f} |  |  |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Full experiment suite (CLI)
# --------------------------------------------------------------------------- #
def run_all(cfg: Config, fracs=(0.1, 0.25, 0.5, 1.0), lengths=(64, 128, 256), out_dir=OUT_DIR):
    set_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    data = load_imdb(cfg)
    rows = []

    main = run("pretrained", "fine-tuning", data, tok, cfg, "main")
    rows.append(main.row)
    frozen = run("pretrained", "frozen", data, tok, cfg, "exp2_frozen_vs_ft")
    rows.append(frozen.row)
    scratch = run("scratch", "from scratch", data, tok, cfg, "exp1_scratch_vs_pretrained")
    rows.append(scratch.row)
    for frac in fracs:
        if frac < 1.0:
            rows.append(run("pretrained", "fine-tuning", data, tok, cfg, "exp3_data", frac).row)
            rows.append(run("scratch", "from scratch", data, tok, cfg, "exp3_data", frac).row)
    for L in lengths:
        if L != cfg.max_len:
            rows.append(run("pretrained", "fine-tuning", data, tok, cfg, "exp4_maxlen", max_len=L).row)
    tw_texts, tw_labels = load_tweets(cfg)
    for r in (main, scratch):
        m, _ = evaluate_on(r.model, tok, tw_texts, tw_labels, cfg.max_len)
        rows.append({**r.row, "experiment": "exp5_domain_shift", "eval_set": "TweetEval (OOD)", **m})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    y = data["test"][1]
    plot_confusion(y, main.probs.argmax(1), os.path.join(out_dir, "confusion_matrix.png"))
    plot_confidence(y, main.probs, os.path.join(out_dir, "confidence_analysis.png"))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    args = ap.parse_args()
    cfg = Config()
    if args.quick:
        cfg = Config(n_train=64, n_val=32, n_test=64, n_ood=64, ft_epochs=1,
                     frozen_epochs=2, scratch_epochs=1)
    print("Hardware:", hardware_info())
    print(run_all(cfg).to_string())


if __name__ == "__main__":
    main()
