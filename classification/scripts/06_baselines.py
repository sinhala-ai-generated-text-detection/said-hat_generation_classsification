"""
Model-free baselines: stylometric and TF-IDF logistic regression.

Each is run on two versions of the text: `raw` (Sinhala-HAT as published) and
`normalized` (the training corpus, scripts/01_build_corpus.py).

Stylometric: logistic regression on the 14 stylometric features (core_14), plus the
share of Latin tokens (full_15), plus the reduced 11-feature set with the four
artefact-linked features removed (reduced_11). Note that social-media posts have no
punctuation, so each is one 'sentence' and the sentence-length features carry no
information there.
TF-IDF: word 1-2-grams and character 2-5-grams, C chosen on dev.
All fit on the train split and scored on the same test split as the encoders.

Outputs in results/baselines/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import RESULTS, load_corpus, load_dataset_df
from stat_features import CORE_14, FULL_15, REDUCED, extract_stat_row

OUT = os.path.join(RESULTS, "baselines")
os.makedirs(OUT, exist_ok=True)

FEATURE_SETS = {"core_14": CORE_14, "full_15": FULL_15, "reduced_11": REDUCED}
C_GRID = [0.1, 1, 10, 100]


def load_texts() -> dict:
    raw, norm = load_dataset_df(), load_corpus()
    norm = norm.set_index("text_id").loc[raw.text_id].reset_index()
    return {"raw": raw, "normalized": norm}


# ------------------------------- stylometric -------------------------------
def stylometric_table(df: pd.DataFrame, tok) -> pd.DataFrame:
    rows = [extract_stat_row(t, tokenizer=tok) for t in df.text]
    return pd.concat([df[["text_id", "domain", "label", "split"]].reset_index(drop=True),
                      pd.DataFrame(rows)], axis=1)


def fit_eval(feat: pd.DataFrame, cols: list[str]) -> dict:
    tr, dv, te = (feat[feat.split == s] for s in ("train", "dev", "test"))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    pipe.fit(tr[cols], tr.label)

    def score(sub):
        proba = pipe.predict_proba(sub[cols])[:, 1]
        pred = pipe.predict(sub[cols])
        return {"roc_auc": roc_auc_score(sub.label, proba), "macro_f1": f1_score(sub.label, pred, average="macro"),
                "accuracy": accuracy_score(sub.label, pred)}
    res = {"dev": score(dv), "test": score(te)}
    res["by_domain"] = {d: score(te[te.domain == d]) for d in sorted(te.domain.unique())}
    return res


def run_stylometric(texts):
    rows, dom_rows = [], []
    tok = None
    for name, df in texts.items():
        path = os.path.join(OUT, f"stylometric_features_{name}.parquet")
        if os.path.exists(path):
            feat = pd.read_parquet(path)
            assert set(feat.text_id) == set(df.text_id), f"{path} is out of date; delete it"
            feat = feat.set_index("text_id").loc[df.text_id].reset_index()
        else:
            if tok is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
            print(f"[stylometric] {name}: extracting features...", flush=True)
            feat = stylometric_table(df, tok)
            feat.to_parquet(path)
        for fs, cols in FEATURE_SETS.items():
            res = fit_eval(feat, cols)
            rows.append({"text": name, "feature_set": fs, "n_features": len(cols),
                         "dev_roc_auc": round(res["dev"]["roc_auc"], 4),
                         "test_roc_auc": round(res["test"]["roc_auc"], 4),
                         "test_macro_f1": round(res["test"]["macro_f1"], 4),
                         "test_accuracy": round(res["test"]["accuracy"], 4)})
            for d, s in res["by_domain"].items():
                dom_rows.append({"text": name, "feature_set": fs, "domain": d,
                                 "test_macro_f1": round(s["macro_f1"], 4)})
            print(f"    {name} / {fs}: test AUC {res['test']['roc_auc']:.4f}  "
                  f"macro-F1 {res['test']['macro_f1']:.4f}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(dom_rows)


# --------------------------------- TF-IDF ----------------------------------
def make_vec(kind):
    if kind == "word_1_2":
        return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                               ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300000)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False,
                           min_df=3, sublinear_tf=True, max_features=300000)


def run_tfidf(texts):
    rows, dom_rows = [], []
    for name, df in texts.items():
        tr, dv, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "dev", "test"))
        for kind in ["word_1_2", "char_2_5"]:
            print(f"[tfidf] {name} / {kind}", flush=True)
            vec = make_vec(kind)
            Xtr = vec.fit_transform(tr.text)
            Xdv, Xte = vec.transform(dv.text), vec.transform(te.text)
            best = None
            for C in C_GRID:
                clf = LogisticRegression(C=C, solver="liblinear", max_iter=1000, random_state=42).fit(Xtr, tr.label)
                f = f1_score(dv.label, clf.predict(Xdv), average="macro")
                if best is None or f > best[0]:
                    best = (f, C, clf)
            dev_f1, C, clf = best
            proba, pred = clf.predict_proba(Xte)[:, 1], clf.predict(Xte)
            rows.append({"text": name, "features": kind, "C": C, "dev_macro_f1": round(dev_f1, 4),
                         "test_macro_f1": round(f1_score(te.label, pred, average="macro"), 4),
                         "test_accuracy": round(accuracy_score(te.label, pred), 4),
                         "test_roc_auc": round(roc_auc_score(te.label, proba), 4)})
            for d in sorted(te.domain.unique()):
                m = (te.domain == d).values
                dom_rows.append({"text": name, "features": kind, "domain": d,
                                 "test_macro_f1": round(f1_score(te.label[m], pred[m], average="macro"), 4)})
            print(f"    C={C} test macro-F1 {rows[-1]['test_macro_f1']}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(dom_rows)


def main():
    texts = load_texts()

    sty, sty_dom = run_stylometric(texts)
    sty.to_csv(os.path.join(OUT, "stylometric_summary.csv"), index=False)
    sty_dom.to_csv(os.path.join(OUT, "stylometric_by_domain.csv"), index=False)

    tf, tf_dom = run_tfidf(texts)
    tf.to_csv(os.path.join(OUT, "tfidf_summary.csv"), index=False)
    tf_dom.to_csv(os.path.join(OUT, "tfidf_by_domain.csv"), index=False)

    lines = ["# Stylometric and TF-IDF baselines\n",
             "`raw` is Sinhala-HAT as published; `normalized` is the encoders' training corpus. "
             "Same train/dev/test split as the encoders.\n",
             "## Stylometric logistic regression\n", sty.to_markdown(index=False), "",
             "\n### Stylometric, full_15, test macro-F1 by domain\n",
             sty_dom[sty_dom.feature_set == "full_15"].pivot_table(
                 index="text", columns="domain", values="test_macro_f1").to_markdown(), "",
             "\n## TF-IDF logistic regression\n", tf.to_markdown(index=False), "",
             "\n### TF-IDF, test macro-F1 by domain\n",
             tf_dom.pivot_table(index=["features", "text"], columns="domain", values="test_macro_f1").to_markdown(), ""]
    with open(os.path.join(OUT, "BASELINES.md"), "w") as f:
        f.write("\n".join(lines))
    print(sty.to_string(index=False))
    print(tf.to_string(index=False))
    print("\nDONE ->", os.path.join(OUT, "BASELINES.md"))


if __name__ == "__main__":
    main()
