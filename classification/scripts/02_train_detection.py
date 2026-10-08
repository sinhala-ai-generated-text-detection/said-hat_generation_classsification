"""
Binary detection: fine-tune four encoders in-distribution and leave-one-domain-out.

  (a) In-distribution, all three domains: LR search over {2, 3, 5}e-5 at seed 42
      (selected on dev macro-F1), then 3 seeds (42, 123, 2026) at the best LR.
  (b) Leave-one-domain-out (LODO): hold out every row of one domain, re-split the
      other two 75/25 by pair (seed 42), single seed (42) at the in-distribution LR.

Analysis: 3-seed mean, seed std and pair-clustered bootstrap 95% CI (5,000 resamples)
on the test split, per-domain scores, and the LODO table. Each LODO fold is compared
with the in-distribution score on the same domain's test slice.

Resumable. Pass --analyse-only to rebuild the tables from saved runs without training.
Outputs in results/detection/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data_utils import (MODELS, RESULTS, holdout_split, load_corpus, load_dataset_df, load_runs,
                        pred_path, splits)
from stats_utils import ci95, pair_members, paired_bootstrap

OUT = os.path.join(RESULTS, "detection")
os.makedirs(OUT, exist_ok=True)
SEED = 42
BUDGET_PATH = os.path.join(OUT, "max_length_by_model.json")
BEST_LR_PATH = os.path.join(OUT, "best_lr.json")


def get_budget():
    if os.path.exists(BUDGET_PATH):
        return json.load(open(BUDGET_PATH))
    from train import compute_max_length_by_model
    budget = compute_max_length_by_model(MODELS, load_dataset_df())
    json.dump(budget, open(BUDGET_PATH, "w"), indent=2)
    return budget


def run_indist(df, budget):
    from train import Engine, LR_CANDIDATES, SEEDS_MAIN, SEED_SEARCH
    engine = Engine(OUT, budget, models=MODELS)
    tr, dv, te = splits(df)
    print(f"IN-DISTRIBUTION  train {len(tr)} / dev {len(dv)} / test {len(te)}", flush=True)
    for nice in MODELS:
        print(f"\n--- LR search: {nice} ---", flush=True)
        for lr in LR_CANDIDATES:
            r, _ = engine.run(nice, tr, dv, te, seed=SEED_SEARCH, lr=lr, tag="search")
            print(f"    lr={lr} dev={r['dev_macro_f1']:.4f} test={r['macro_f1']:.4f} "
                  f"epoch={r['epochs_used']} ({r['minutes']} min)", flush=True)
        s = engine.load_runs("search")
        s = s[s.Model == nice]
        best_lr = float(s.loc[s.dev_macro_f1.idxmax()].lr)
        allbest = json.load(open(BEST_LR_PATH)) if os.path.exists(BEST_LR_PATH) else {}
        allbest[nice] = best_lr
        json.dump(allbest, open(BEST_LR_PATH, "w"), indent=2)
        print(f"    best_lr({nice}) = {best_lr}", flush=True)
        for seed in SEEDS_MAIN:
            r, _ = engine.run(nice, tr, dv, te, seed=seed, lr=best_lr, tag="main")
            print(f"    main seed={seed} dev={r['dev_macro_f1']:.4f} test={r['macro_f1']:.4f} "
                  f"epoch={r['epochs_used']} ({r['minutes']} min)", flush=True)


def run_lodo(df, budget):
    from train import Engine
    engine = Engine(OUT, budget, models=MODELS)
    best_lr = json.load(open(BEST_LR_PATH))
    for nice in MODELS:
        lr = best_lr[nice]
        for dom in sorted(df.domain.unique()):
            seen = df[df.domain != dom]
            te = df[df.domain == dom].reset_index(drop=True)
            tr, dv = holdout_split(seen, seed=SEED)
            assert dom not in set(tr.domain) | set(dv.domain), "held-out domain leaked into training"
            assert not (set(tr.group) | set(dv.group)) & set(te.group), "pair leaked into test"
            print(f"\n--- LODO | {nice} | hold out {dom} | lr={lr} "
                  f"| train {len(tr)} / dev {len(dv)} / test {len(te)} ---", flush=True)
            r, _ = engine.run(nice, tr, dv, te, seed=SEED, lr=lr, tag="lodo", note=dom)
            print(f"    unseen-{dom}: macro-F1 {r['macro_f1']:.4f} ({r['minutes']} min)", flush=True)


def main_preds(model):
    r = load_runs(OUT, "main")
    r = r[r.Model == model].sort_values("seed")
    return np.stack([np.load(pred_path(OUT, x)) for x in r.run_id]), r


def analyse(df):
    te = splits(df)[2]
    y = te.label.values
    members = pair_members(te.group.values)

    overall, by_domain, lodo_rows = [], [], []
    lodo = load_runs(OUT, "lodo")
    for model in MODELS:
        preds, rows = main_preds(model)
        boot = paired_bootstrap(y, {"m": preds}, members, n_boot=5000, seed=0)["m"]["boot"]
        lo, hi = ci95(boot)
        seed_std = rows.macro_f1.std()
        overall.append({"Model": model, "lr": float(rows.lr.iloc[0]), "n_seeds": len(rows),
                        "test_macro_f1_mean": round(rows.macro_f1.mean(), 4), "seed_std": round(seed_std, 4),
                        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                        "dev_macro_f1_mean": round(rows.dev_macro_f1.mean(), 4)})
        for dom in sorted(te.domain.unique()):
            m = (te.domain == dom).values
            scores = [f1_score(y[m], p[m], average="macro") for p in preds]
            by_domain.append({"Model": model, "domain": dom, "macro_f1_mean": round(float(np.mean(scores)), 4),
                              "seed_std": round(float(np.std(scores, ddof=1)), 4)})
            r = lodo[(lodo.Model == model) & (lodo.note == dom)]
            if len(r):
                indist = float(np.mean(scores))
                thr = 2 * max(float(np.std(scores, ddof=1)), seed_std)
                drop = indist - float(r.macro_f1.iloc[0])
                lodo_rows.append({"Model": model, "held_out": dom, "lr": float(r.lr.iloc[0]),
                                  "holdout_f1": round(float(r.macro_f1.iloc[0]), 4),
                                  "indist_same_slice": round(indist, 4), "drop": round(drop, 4),
                                  "threshold": round(thr, 4), "real": "yes" if drop > thr else "noise"})

    overall, by_domain, lodo_tbl = pd.DataFrame(overall), pd.DataFrame(by_domain), pd.DataFrame(lodo_rows)
    overall.to_csv(os.path.join(OUT, "indist_summary.csv"), index=False)
    by_domain.to_csv(os.path.join(OUT, "indist_by_domain.csv"), index=False)
    lodo_tbl.to_csv(os.path.join(OUT, "lodo_summary.csv"), index=False)

    lines = ["# Binary detection — in-distribution and leave-one-domain-out\n",
             "Normalized Sinhala-HAT, test split (675 pairs). In-distribution: mean and seed std over "
             "3 seeds at the dev-selected LR; 95% CI from a pair-clustered bootstrap (5,000 resamples, "
             "test-set sampling only).\n",
             "## In-distribution: test macro-F1\n", overall.to_markdown(index=False), "",
             "\n## In-distribution: by domain\n",
             by_domain.pivot_table(index="Model", columns="domain", values="macro_f1_mean").round(4).to_markdown(), "",
             "\n## Leave-one-domain-out (single seed 42)\n",
             "`indist_same_slice` is the in-distribution 3-seed score on the held-out domain's test slice. "
             "`real` marks a drop larger than twice the larger of that slice's seed spread and the "
             "model's overall seed std.\n",
             lodo_tbl.to_markdown(index=False) if len(lodo_tbl) else "*(no LODO runs yet)*", ""]
    with open(os.path.join(OUT, "DETECTION.md"), "w") as f:
        f.write("\n".join(lines))
    return overall, lodo_tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true", help="rebuild tables from saved runs, no training")
    args = ap.parse_args()

    df = load_corpus()
    if not args.analyse_only:
        budget = get_budget()
        print("budget:", json.dumps(budget, indent=2))
        run_indist(df, budget)
        run_lodo(df, budget)
    overall, lodo_tbl = analyse(df)
    print(overall.to_string(index=False))
    print(lodo_tbl.to_string(index=False))
    print("\nDONE ->", os.path.join(OUT, "DETECTION.md"))


if __name__ == "__main__":
    main()
