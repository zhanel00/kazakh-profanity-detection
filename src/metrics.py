"""Metric helpers used by training (validation) and evaluation (test)."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def binary_report(y_true, prob, threshold):
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "precision_profane": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_profane": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_profane": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true, prob, grid=np.arange(0.20, 0.801, 0.01)):
    """Threshold that maximises F1 of the profanity class on VALIDATION data.

    The search is limited to [0.20, 0.80]: the validation set has only 265
    texts, and a fine-tuned transformer puts most probabilities near 0 or 1, so
    an extreme threshold (e.g. 0.08) would be fitted to 2-3 borderline examples
    rather than to a real pattern. Ties are broken towards 0.5."""
    scores = [(f1_score(y_true, (prob >= t).astype(int), zero_division=0), -abs(t - 0.5), t)
              for t in grid]
    f1, _, t = max(scores)
    return float(round(t, 2)), float(f1)
