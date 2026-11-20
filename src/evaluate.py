"""Step 3 - ONE final evaluation on the held-out test set.

    python -m src.evaluate                         # local model in artifacts/model
    python -m src.evaluate --model <hf-user>/kazakh-profanity-xlmr

Outputs (results/): test_metrics.json, confusion_matrix.png,
                    test_predictions.csv, test_errors.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import binary_report  # noqa: E402
from .model_utils import load_model, predict_proba  # noqa: E402


def plot_cm(cm, path, title):
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=14,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred 0\n(clean)", "pred 1\n(profane)"])
    ax.set_yticks([0, 1], ["true 0\n(clean)", "true 1\n(profane)"])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--model", help="local dir or HF Hub id (default: config output_dir)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    tok, model, s, device = load_model(args.model or cfg["output_dir"])
    df = pd.read_csv(Path(cfg["processed_dir"]) / f"{args.split}.csv")
    prob = predict_proba(df.text.tolist(), tok, model, device,
                         s["max_length"], s["window_stride"], cfg["eval_batch_size"])
    rep = binary_report(df.label.values, prob, s["threshold"])
    rep["also_at_0.5"] = binary_report(df.label.values, prob, 0.5)
    print(json.dumps(rep, indent=2))

    res = Path(cfg["results_dir"])
    res.mkdir(exist_ok=True)
    (res / f"{args.split}_metrics.json").write_text(json.dumps(rep, indent=2))
    plot_cm(rep["confusion_matrix"], res / f"confusion_matrix_{args.split}.png",
            f"{args.split} set (n={rep['n']}), threshold={s['threshold']}")

    df["prob_profane"] = prob.round(4)
    df["pred"] = (prob >= s["threshold"]).astype(int)
    df["error"] = np.select([(df.pred == 1) & (df.label == 0), (df.pred == 0) & (df.label == 1)],
                            ["FP", "FN"], "")
    df.drop(columns=["group"]).to_csv(res / f"{args.split}_predictions.csv", index=False)
    err = df[df.error != ""].sort_values("prob_profane")
    err.drop(columns=["group"]).to_csv(res / f"{args.split}_errors.csv", index=False)
    print(f"\n{len(err)} errors:")
    for r in err.itertuples():
        print(f"[{r.error}] p={r.prob_profane:.3f} | {r.text[:120]}")


if __name__ == "__main__":
    main()
