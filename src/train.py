"""Step 2 - fine-tune XLM-RoBERTa-base for binary profanity classification.

    python -m src.train                 # uses config.json
    python -m src.train --epochs 3      # quick override

* training objective: cross-entropy over 2 classes (0 = clean, 1 = profane)
* model selection: the epoch with the best validation F1 (profane class)
* decision threshold: tuned on validation after training (never on test)
Outputs: artifacts/model/ (weights, tokenizer, inference_config.json)
         results/training_log.csv, results/val_metrics.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from .metrics import best_threshold, binary_report
from .model_utils import get_device, predict_proba, set_seed
from .text_utils import normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--output_dir", help="override (e.g. for a hyper-parameter sweep)")
    ap.add_argument("--results_dir", help="override (e.g. for a hyper-parameter sweep)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.epochs:
        cfg["num_epochs"] = args.epochs
    if args.lr:
        cfg["learning_rate"] = args.lr
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.results_dir:
        cfg["results_dir"] = args.results_dir

    set_seed(cfg["seed"])
    device = get_device()
    print("device:", device)

    data = Path(cfg["processed_dir"])
    train_df = pd.read_csv(data / "train.csv")
    val_df = pd.read_csv(data / "val.csv")
    print(f"train={len(train_df)}  val={len(val_df)}")

    # ---- text -> model input --------------------------------------------------
    # normalize() (rule-based) -> SentencePiece tokenizer of XLM-R (pre-trained,
    # NOT re-fitted here, so nothing is learned from val/test). Unknown words are
    # split into known sub-word pieces; truly unknown symbols map to <unk>.
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    enc = tok([normalize(t) for t in train_df.text], truncation=True,
              max_length=cfg["max_length"])
    train_items = [{"input_ids": ids, "attention_mask": am, "labels": int(y)}
                   for ids, am, y in zip(enc["input_ids"], enc["attention_mask"],
                                         train_df.label)]
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train_items, batch_size=cfg["batch_size"], shuffle=True,
                        generator=g, collate_fn=DataCollatorWithPadding(tok))

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2,
        id2label={0: "clean", 1: "profane"}, label2id={"clean": 0, "profane": 1})
    model.to(device)

    # The word-embedding matrix (250k x 768 = 192M of XLM-R's 278M parameters)
    # is frozen: it removes ~2.3 GB of gradients + Adam state (the model fits in
    # 16 GB RAM) and, with only 2.5k training texts, limits overfitting. All 12
    # transformer layers and the classification head are still fine-tuned.
    if cfg.get("freeze_word_embeddings", False):
        model.roberta.embeddings.word_embeddings.weight.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train/1e6:.1f}M of {n_all/1e6:.1f}M")

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [
        {"params": [p for n, p in named if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in named if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=cfg["learning_rate"])
    total = len(loader) * cfg["num_epochs"]
    sched = get_linear_schedule_with_warmup(optim, int(cfg["warmup_ratio"] * total), total)

    out_dir = Path(cfg["output_dir"])
    res_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    log, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss          # cross-entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optim.step()
            sched.step()
            optim.zero_grad()
            run_loss += loss.item()
            if step % 40 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} "
                      f"loss={run_loss/step:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
        model.eval()
        val_prob = predict_proba(val_df.text.tolist(), tok, model, device,
                                 cfg["max_length"], cfg["window_stride"], cfg["eval_batch_size"])
        val_loss = float(np.mean(-np.log(np.clip(
            np.where(val_df.label == 1, val_prob, 1 - val_prob), 1e-7, 1))))
        rep = binary_report(val_df.label.values, val_prob, 0.5)
        row = {"epoch": epoch, "train_loss": run_loss / len(loader), "val_loss": val_loss,
               "val_f1@0.5": rep["f1_profane"], "val_precision@0.5": rep["precision_profane"],
               "val_recall@0.5": rep["recall_profane"], "seconds": round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps(row))

        if rep["f1_profane"] > best_f1:
            best_f1, bad_epochs = rep["f1_profane"], 0
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            print(f"  -> new best (val F1={best_f1:.4f}), saved to {out_dir}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stopping_patience"]:
                print("early stopping")
                break

    pd.DataFrame(log).to_csv(res_dir / "training_log.csv", index=False)

    # ---- threshold selection on validation ----------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device).eval()
    val_prob = predict_proba(val_df.text.tolist(), tok, model, device,
                             cfg["max_length"], cfg["window_stride"], cfg["eval_batch_size"])
    thr, thr_f1 = best_threshold(val_df.label.values, val_prob)
    best_epoch = max(log, key=lambda r: r["val_f1@0.5"])["epoch"]
    print(f"best epoch={best_epoch}; tuned threshold={thr} (val F1={thr_f1:.4f})")

    inference_cfg = {"threshold": thr, "max_length": cfg["max_length"],
                     "window_stride": cfg["window_stride"], "base_model": cfg["model_name"],
                     "best_epoch": best_epoch, "seed": cfg["seed"]}
    (out_dir / "inference_config.json").write_text(json.dumps(inference_cfg, indent=2))
    val_metrics = {"at_0.5": binary_report(val_df.label.values, val_prob, 0.5),
                   "at_tuned_threshold": binary_report(val_df.label.values, val_prob, thr),
                   "config": cfg, "best_epoch": best_epoch}
    (res_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))


if __name__ == "__main__":
    main()
