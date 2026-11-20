"""Re-select the decision threshold on VALIDATION data without retraining.

    python -m src.tune_threshold --model artifacts/model --results_dir results

Updates <model>/inference_config.json and <results_dir>/val_metrics.json.
(src.train already does this at the end of training; this script is for
re-tuning after changing the threshold search in src/metrics.py.)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import best_threshold, binary_report
from .model_utils import load_model, predict_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--model", default="artifacts/model")
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    tok, model, s, device = load_model(args.model)
    val = pd.read_csv(Path(cfg["processed_dir"]) / "val.csv")
    prob = predict_proba(val.text.tolist(), tok, model, device,
                         s["max_length"], s["window_stride"], cfg["eval_batch_size"])
    thr, f1 = best_threshold(val.label.values, prob)
    print(f"threshold={thr}  val F1={f1:.4f}")

    p = Path(args.model) / "inference_config.json"
    ic = json.loads(p.read_text())
    ic["threshold"] = thr
    p.write_text(json.dumps(ic, indent=2))

    vm_path = Path(args.results_dir) / "val_metrics.json"
    vm = json.loads(vm_path.read_text()) if vm_path.exists() else {}
    vm["at_0.5"] = binary_report(val.label.values, prob, 0.5)
    vm["at_tuned_threshold"] = binary_report(val.label.values, prob, thr)
    vm_path.write_text(json.dumps(vm, indent=2))


if __name__ == "__main__":
    main()
