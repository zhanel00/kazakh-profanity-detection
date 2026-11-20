"""Step 0 - find suspicious labels for MANUAL review (label audit).

    python -m src.audit_labels                       # counts only
    python -m src.audit_labels --show 0hit           # label 0 but a profane stem occurs
    python -m src.audit_labels --show 1miss          # label 1 but no stem occurs
    python -m src.audit_labels --out audit.csv       # all candidates to a CSV (do not commit:
                                                     # it contains raw, non-anonymised text)

The lexicon below is a hand-compiled list of common Kazakh and Russian obscene stems,
strong slurs and their spelling variants (Latin script, Russian keyboard, distortions),
prepared with AI assistance. It is deliberately broad (it also contains ambiguous words
such as мал, ит, шайтан), because it only PRIORITISES rows for a human to read:

* it is NOT a model feature and is not used in training or inference;
* a label is changed only after reading the text against the labelling rules in the
  README; every change and its reason is recorded in data/label_revisions.csv, which
  src.prepare_data then applies.

On the raw course file this flags 157 label-0 rows with a stem and 246 label-1 rows
without one; reading them gave the 19 revisions.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

STEMS = [
    # Kazakh obscene vocabulary (+ Russian-keyboard and Latin spellings)
    "қотақ", "котак", "қотаг", "котаг", "qotaq", "kotak",
    "амы", "ам", "амын", "амына",
    "сік", "сиг", "сіг", "сик", "сек", "sik", "sig", "сиктир", "сіктір",
    "шешең", "шешен", "шешес", "шеше", "sheshe",
    "көтен", "котен", "кот", "көт", "kote",
    "қаншық", "канщык", "қаншык", "жалап", "zhalap", "қылкен", "кылкен",
    # strong slurs / insults (ambiguous: read in context)
    "мал", "mal", "есек", "далбас", "ақымақ", "акымак", "жынды", "шайтан", "иттің", "итин", "ит ",
    # Russian mat and insults inside Kazakh comments
    "бля", "блә", "bla", "наху", "нах", "nahu", "хуй", "хуя", "хуе", "xuy", "huy",
    "пизд", "писд", "pizd", "ебал", "еба", "ебу", "заеб", "заиб", "зайбал",
    "сука", "сучк", "suka", "говн", "гавн", "gavn", "govn", "гандон", "долбо", "мудак",
    "пидор", "пидр", "шлюх", "чмо", "урод", "тупо", "дебил", "идиот",
]

# a stem must start a token (no letter directly before it)
_TOKEN_START = r"(?<![^\W\d_])"
LEXICON = re.compile(_TOKEN_START + "(" + "|".join(
    sorted(map(re.escape, STEMS), key=len, reverse=True)) + ")")


def stem_hits(text: str) -> list:
    return sorted(set(LEXICON.findall(str(text).lower())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--raw", help="raw CSV (default: raw_csv from config)")
    ap.add_argument("--show", choices=["0hit", "1miss"])
    ap.add_argument("--out", help="write all candidates to this CSV")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    df = pd.read_csv(args.raw or cfg["raw_csv"])

    df["stem_hits"] = df.text.map(stem_hits)
    has = df.stem_hits.str.len() > 0
    groups = {"0hit": df[(df.label == 0) & has], "1miss": df[(df.label == 1) & ~has]}
    print(f"{len(STEMS)} stems | rows={len(df)}")
    print(f"label 0 with a stem   (0hit):  {len(groups['0hit'])}")
    print(f"label 1 without stem (1miss): {len(groups['1miss'])}")

    if args.show:
        for _, r in groups[args.show].iterrows():
            print(r.kazsandra_id, r.split, r.stem_hits, "|", str(r.text).replace("\n", " ")[:200])
    if args.out:
        out = pd.concat([g.assign(audit_group=k) for k, g in groups.items()])
        out[["kazsandra_id", "label", "split", "audit_group", "stem_hits", "text"]].to_csv(
            args.out, index=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
