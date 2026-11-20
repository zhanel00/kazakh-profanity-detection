"""Shared model helpers: seeding, device choice, loading, and windowed prediction."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .text_utils import normalize


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir_or_hub_id, device=None):
    """Load tokenizer + fine-tuned model + inference settings (threshold, window).

    `model_dir_or_hub_id` may be a local folder (artifacts/model) or a
    Hugging Face Hub repo id (e.g. "<user>/kazakh-profanity-xlmr")."""
    device = device or get_device()
    tok = AutoTokenizer.from_pretrained(model_dir_or_hub_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir_or_hub_id)
    model.to(device).eval()
    settings = {"threshold": 0.5, "max_length": 256, "window_stride": 128}
    local = Path(model_dir_or_hub_id) / "inference_config.json"
    if local.exists():
        settings.update(json.loads(local.read_text()))
    else:  # try the Hub
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(model_dir_or_hub_id, "inference_config.json")
            settings.update(json.loads(Path(p).read_text()))
        except Exception:
            print("inference_config.json not found - using threshold 0.5")
    return tok, model, settings, device


@torch.no_grad()
def predict_proba(texts, tok, model, device, max_length=256, stride=128, batch_size=64):
    """Return P(profanity) for each text.

    Long comments: the tokenizer cuts a text into overlapping windows of
    `max_length` tokens (overlap = `stride`). Each window is classified and the
    text gets the MAX window probability - one profane window is enough to
    flag the whole comment."""
    texts = [normalize(t) for t in texts]
    enc = tok(texts, truncation=True, max_length=max_length, stride=stride,
              return_overflowing_tokens=True, padding=False)
    owner = enc["overflow_to_sample_mapping"]
    probs_per_window = []
    for i in range(0, len(enc["input_ids"]), batch_size):
        batch = tok.pad({"input_ids": enc["input_ids"][i:i + batch_size],
                         "attention_mask": enc["attention_mask"][i:i + batch_size]},
                        return_tensors="pt")
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        probs_per_window.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().tolist())
    out = np.zeros(len(texts))
    for p, o in zip(probs_per_window, owner):
        out[o] = max(out[o], p)
    return out
