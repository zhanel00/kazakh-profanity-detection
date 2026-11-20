"""Reference only (NOT the submitted model): char n-gram TF-IDF + logistic regression.

    python -m src.baseline

Shows how much the fine-tuned transformer adds over a purely lexical model.
The TF-IDF vocabulary is fitted on the TRAIN split only.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .metrics import best_threshold, binary_report
from .text_utils import normalize


def main():
    cfg = json.loads(Path("config.json").read_text())
    d = Path(cfg["processed_dir"])
    tr, va, te = (pd.read_csv(d / f"{s}.csv") for s in ("train", "val", "test"))
    clf = make_pipeline(
        TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                        min_df=2, sublinear_tf=True),
        LogisticRegression(C=10, max_iter=2000, random_state=cfg["seed"]))
    clf.fit(tr.text, tr.label)
    pv = clf.predict_proba(va.text)[:, 1]
    thr, _ = best_threshold(va.label.values, pv)
    res = {"model": "char(2-5) TF-IDF + LogisticRegression(C=10)",
           "val": binary_report(va.label.values, pv, thr),
           "test": binary_report(te.label.values, clf.predict_proba(te.text)[:, 1], thr)}
    print(json.dumps(res, indent=2))
    Path(cfg["results_dir"]).mkdir(exist_ok=True)
    (Path(cfg["results_dir"]) / "baseline_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
