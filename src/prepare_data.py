"""Step 1 - build clean, leak-free train/val/test files from the raw CSV.

    python -m src.prepare_data            # keeps the provided split, repairs leaks
    python -m src.prepare_data --resplit  # fresh 80/10/10 stratified group split

Pipeline
  1. anonymise text (e-mails, URLs, phones, handles, surnames)
  2. apply the documented label revisions (data/label_revisions.csv)
  3. drop exact duplicates (after aggressive normalisation)
  4. group near-duplicates (char 3-5-gram TF-IDF cosine >= threshold, union-find)
  5. move every near-duplicate group into ONE split
  6. save data/processed/{train,val,test}.csv + split_report.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from .text_utils import anonymize, dedup_key

SPLITS = ["train", "val", "test"]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def near_duplicate_groups(keys, threshold):
    """Return a group id per row. Rows whose dedup keys have char-n-gram cosine
    similarity >= threshold end up in the same group (transitively).
    NOTE: this TF-IDF is a data-cleaning tool only - it is NOT a model feature."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    X = vec.fit_transform(keys)
    nn = NearestNeighbors(radius=1 - threshold, metric="cosine").fit(X)
    _, neigh = nn.radius_neighbors(X)
    uf = UnionFind(len(keys))
    for i, js in enumerate(neigh):
        for j in js:
            uf.union(i, int(j))
    return np.array([uf.find(i) for i in range(len(keys))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--resplit", action="store_true",
                    help="ignore the provided split column and create a new 80/10/10 split")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    report = {}

    df = pd.read_csv(cfg["raw_csv"], encoding="utf-8-sig")
    df = df.rename(columns={"kazsandra_id": "id"})
    report["raw_rows"] = len(df)
    report["raw_label_counts"] = df.label.value_counts().sort_index().to_dict()

    # 1. anonymise --------------------------------------------------------------
    anon = df.text.map(anonymize)
    report["rows_anonymised"] = int((anon != df.text).sum())
    df["text"] = anon

    # 2. label revisions -------------------------------------------------------
    rev = pd.read_csv(cfg["label_revisions"]).rename(columns={"kazsandra_id": "id"})
    df = df.set_index("id")
    for r in rev.itertuples():
        assert df.at[r.id, "label"] == r.old_label, f"unexpected label for {r.id}"
        df.at[r.id, "label"] = r.new_label
    df = df.reset_index()
    report["labels_revised"] = {"0->1": int((rev.new_label == 1).sum()),
                                "1->0": int((rev.new_label == 0).sum())}

    # 3. exact duplicates ------------------------------------------------------
    df["key"] = df.text.map(dedup_key)
    df = df[df.key.str.len() > 0]
    conflicts = df.groupby("key").label.nunique()
    conflicts = conflicts[conflicts > 1].index.tolist()
    if conflicts:
        print("WARNING - duplicates with conflicting labels:", conflicts)
    before = len(df)
    df = df.drop_duplicates("key", keep="first").reset_index(drop=True)
    report["exact_duplicates_removed"] = before - len(df)

    # 4. near-duplicate groups -------------------------------------------------
    df["group"] = near_duplicate_groups(df.key.tolist(), cfg["near_duplicate_threshold"])
    sizes = df.group.value_counts()
    report["near_duplicate_groups(size>1)"] = int((sizes > 1).sum())
    report["rows_in_near_duplicate_groups"] = int(sizes[sizes > 1].sum())

    # 5. assign splits ---------------------------------------------------------
    if args.resplit:
        rng = cfg["seed"]
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=rng)
        folds = np.zeros(len(df), dtype=int)
        for k, (_, idx) in enumerate(sgkf.split(df, df.label, df.group)):
            folds[idx] = k
        df["split"] = np.where(folds == 0, "test", np.where(folds == 1, "val", "train"))
        report["split_strategy"] = f"new StratifiedGroupKFold 80/10/10, seed={rng}"
    else:
        moved = 0
        for g, idx in df.groupby("group").groups.items():
            splits = df.loc[idx, "split"]
            if splits.nunique() > 1:
                cnt = Counter(splits)
                # majority split; ties broken towards train (keeps test conservative)
                target = max(SPLITS, key=lambda s: (cnt.get(s, 0), -SPLITS.index(s)))
                moved += int((splits != target).sum())
                df.loc[idx, "split"] = target
        report["split_strategy"] = ("provided split column, near-duplicate groups "
                                    "moved to their majority split")
        report["rows_moved_to_fix_leakage"] = moved

    # sanity: no group may span two splits
    assert df.groupby("group").split.nunique().max() == 1

    # 6. save ------------------------------------------------------------------
    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    report["final_rows"] = len(df)
    report["splits"] = {}
    for s in SPLITS:
        part = df[df.split == s][["id", "text", "label", "group"]]
        part.to_csv(out / f"{s}.csv", index=False)
        vc = part.label.value_counts().to_dict()
        report["splits"][s] = {"n": len(part), "label_0": int(vc.get(0, 0)),
                               "label_1": int(vc.get(1, 0)),
                               "profane_share": round(float(part.label.mean()), 3)}
    report["seed"] = cfg["seed"]
    (out / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
