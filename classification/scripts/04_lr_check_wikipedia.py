"""
Learning-rate check on a held-out Wikipedia fold, XLM-R base.

Hold out every Wikipedia row as the test set, re-split news + social media 75/25 by
pair (seed 42) into train/dev, and train once at each candidate LR (seed 42). Reports
dev macro-F1 on the fold's own dev split and test macro-F1 on unseen Wikipedia, to show
whether dev-based LR selection also picks a rate that transfers.

Needs the window budget from scripts/02_train_detection.py. Resumable; --analyse-only
rebuilds the table from saved runs without training.
Outputs in results/lr_check/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from data_utils import MODELS, RESULTS, holdout_split, load_corpus, load_runs

DET = os.path.join(RESULTS, "detection")
OUT = os.path.join(RESULTS, "lr_check")
os.makedirs(OUT, exist_ok=True)

CHECK_MODEL = "XLM-R base"
CHECK_FOLD = "wikipedia"
SEED = 42


def train():
    from train import Engine, LR_CANDIDATES

    df = load_corpus()
    te = df[df.domain == CHECK_FOLD].reset_index(drop=True)
    seen = df[df.domain != CHECK_FOLD]
    tr, dv = holdout_split(seen, seed=SEED)
    assert CHECK_FOLD not in set(tr.domain) | set(dv.domain), "held-out fold leaked into training"
    assert not (set(tr.group) | set(dv.group)) & set(te.group), "pair leaked into test"
    print(f"seen (news + social_media): {len(seen)} -> train {len(tr)} / dev {len(dv)} | test {len(te)}", flush=True)

    budget = json.load(open(os.path.join(DET, "max_length_by_model.json")))
    engine = Engine(OUT, budget, models={CHECK_MODEL: MODELS[CHECK_MODEL]})
    for lr in LR_CANDIDATES:
        print(f"\n--- lr check: {CHECK_MODEL} | hold out {CHECK_FOLD} | lr {lr} ---", flush=True)
        r, _ = engine.run(CHECK_MODEL, tr, dv, te, seed=SEED, lr=lr, tag="lrcheck", note=CHECK_FOLD)
        print(f"    dev {r['dev_macro_f1']:.4f} | unseen-{CHECK_FOLD} {r['macro_f1']:.4f} ({r['minutes']} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true", help="rebuild the table from saved runs, no training")
    if not ap.parse_args().analyse_only:
        train()

    runs = load_runs(OUT, "lrcheck").sort_values("lr")
    rows = [{"Model": r.Model, "held_out": r.note, "lr": r.lr, "dev_macro_f1": round(r.dev_macro_f1, 4),
             "test_macro_f1": round(r.macro_f1, 4)} for r in runs.itertuples()]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "lr_check.csv"), index=False)

    dev_spread = table.dev_macro_f1.max() - table.dev_macro_f1.min()
    best = table.loc[table.dev_macro_f1.idxmax()]
    lines = ["# Learning-rate check on a held-out Wikipedia fold\n",
             f"{CHECK_MODEL}, trained on news + social media (75/25 pair-grouped re-split, seed 42), "
             f"one run per rate — read as point estimates, not means.\n",
             table.to_markdown(index=False), "",
             f"\nDev macro-F1 spread across the three rates: {dev_spread:.4f}. "
             f"Best rate on this fold's own dev: {best.lr} (dev {best.dev_macro_f1:.4f}, "
             f"unseen-{CHECK_FOLD} test {best.test_macro_f1:.4f}).\n"]
    with open(os.path.join(OUT, "LR_CHECK.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n" + table.to_string(index=False))
    print("\nDONE ->", os.path.join(OUT, "LR_CHECK.md"))


if __name__ == "__main__":
    main()
