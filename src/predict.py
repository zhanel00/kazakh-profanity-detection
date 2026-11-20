"""Predict on new Kazakh comments with the saved model (no retraining).

    python -m src.predict "Өте керемет ойын, ұнады!"
    python -m src.predict "text 1" "text 2" --model <hf-user>/kazakh-profanity-xlmr
    python -m src.predict                   # interactive mode, empty line to quit

In Python / a notebook:
    from src.predict import Predictor
    clf = Predictor("artifacts/model")
    clf("Қотақбас ойын")   # -> {'label': 1, 'prob_profane': 0.99, ...}
"""
import argparse

from .model_utils import load_model, predict_proba


class Predictor:
    def __init__(self, model="artifacts/model"):
        self.tok, self.model, self.s, self.device = load_model(model)

    def __call__(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        probs = predict_proba(texts, self.tok, self.model, self.device,
                              self.s["max_length"], self.s["window_stride"])
        out = [{"text": t, "label": int(p >= self.s["threshold"]),
                "prob_profane": round(float(p), 4), "threshold": self.s["threshold"]}
               for t, p in zip(texts, probs)]
        return out[0] if single else out


def _show(r):
    name = "PROFANE" if r["label"] else "clean"
    print(f"{r['label']} ({name:7s})  P(profane)={r['prob_profane']:.3f}  "
          f"[thr={r['threshold']}]  | {r['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--model", default="artifacts/model",
                    help="local folder or Hugging Face Hub repo id")
    args = ap.parse_args()
    clf = Predictor(args.model)
    if args.texts:
        for r in clf(args.texts):
            _show(r)
        return
    print("Enter a Kazakh comment (empty line to quit):")
    while True:
        try:
            t = input("> ").strip()
        except EOFError:
            break
        if not t:
            break
        _show(clf(t))


if __name__ == "__main__":
    main()
