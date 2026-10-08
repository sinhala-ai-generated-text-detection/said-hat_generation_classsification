"""
Generalisation: leave-one-generator-out and a training-size control.

  (a) Leave-one-generator-out (LOGO), all four encoders: hold out every pair made by
      one generator, re-split the rest 75/25 by pair (seed 42), single seed (42), at
      each model's in-distribution LR (results/detection/best_lr.json).
  (b) Training-size control: train in-distribution on a random 71.5% of the train
      pairs (random_state 42; 4,500 of 6,294 rows), same dev/test, seed 42. A held-out
      fold trains on about 4,500 rows, so this separates the cost of less data from the
      cost of the domain or generator shift.

Needs scripts/02_train_detection.py to have run (LRs, window budget, in-distribution
predictions). Resumable; --analyse-only rebuilds the tables without training.
Outputs in results/generalisation/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data_utils import MODELS, RESULTS, add_pair_generator, holdout_split, load_corpus, load_runs, pred_path, splits

DET = os.path.join(RESULTS, "detection")
OUT = os.path.join(RESULTS, "generalisation")
os.makedirs(OUT, exist_ok=True)
SEED = 42
CTRL_N = 4500


def train_all(df):
    from train import Engine
    budget = json.load(open(os.path.join(DET, "max_length_by_model.json")))
    best_lr = json.load(open(os.path.join(DET, "best_lr.json")))
    engine = Engine(OUT, budget, models=MODELS)
    train_df, dev_df, test_df = splits(df)

    gens = sorted(df[df.label == 1].generator.unique())
    for nice in MODELS:
        lr = best_lr[nice]
        for gen in gens:
            seen = df[df.pair_generator != gen]
            te = df[df.pair_generator == gen].reset_index(drop=True)
            tr, dv = holdout_split(seen, seed=SEED)
            assert gen not in set(tr.pair_generator) | set(dv.pair_generator), "held-out generator leaked"
            assert not (set(tr.group) | set(dv.group)) & set(te.group), "pair leaked into test"
            print(f"\n--- LOGO | {nice} | hold out {gen} | lr={lr} "
                  f"| train {len(tr)} / dev {len(dv)} / test {len(te)} ---", flush=True)
            r, _ = engine.run(nice, tr, dv, te, seed=SEED, lr=lr, tag="logo", note=gen)
            print(f"    unseen-{gen}: macro-F1 {r['macro_f1']:.4f} ({r['minutes']} min)", flush=True)

    groups = pd.Series(train_df.group.unique())
    keep = set(groups.sample(frac=min(1.0, CTRL_N / len(train_df)), random_state=42))
    tr_s = train_df[train_df.group.isin(keep)].reset_index(drop=True)
    print(f"\nsize control: train {len(tr_s)} rows (full {len(train_df)})", flush=True)
    for nice in MODELS:
        lr = best_lr[nice]
        print(f"\n--- size control | {nice} | lr={lr} ---", flush=True)
        r, _ = engine.run(nice, tr_s, dev_df, test_df, seed=SEED, lr=lr, tag="size_control", note=f"n{len(tr_s)}")
        print(f"    test macro-F1 {r['macro_f1']:.4f} ({r['minutes']} min)", flush=True)


def analyse(df):
    runs = load_runs(OUT)
    det_main = load_runs(DET, "main")
    summary = pd.read_csv(os.path.join(DET, "indist_summary.csv")).set_index("Model")
    lodo = pd.read_csv(os.path.join(DET, "lodo_summary.csv"))
    test_df = splits(df)[2]
    y = test_df.label.values

    rows = []
    for _, r in runs[runs.phase == "logo"].sort_values(["Model", "note"]).iterrows():
        gen, model = r.note, r.Model
        mask = (test_df.pair_generator == gen).values
        main = det_main[det_main.Model == model]
        scores = [f1_score(y[mask], np.load(pred_path(DET, x))[mask], average="macro") for x in main.run_id]
        base = float(np.mean(scores))
        thr = 2 * max(float(np.std(scores, ddof=1)), float(summary.loc[model, "seed_std"]))
        rows.append({"Model": model, "held_out_generator": gen, "lr": r.lr,
                     "holdout_f1": round(r.macro_f1, 4), "indist_same_slice": round(base, 4),
                     "drop": round(base - r.macro_f1, 4), "threshold": round(thr, 4),
                     "real": "yes" if (base - r.macro_f1) > thr else "noise"})
    logo_tbl = pd.DataFrame(rows)

    srows = []
    for _, r in runs[runs.phase == "size_control"].sort_values("Model").iterrows():
        model = r.Model
        full_mean = float(summary.loc[model, "test_macro_f1_mean"])
        seed42 = det_main[(det_main.Model == model) & (det_main.seed == 42)]
        lodo_mean = float(lodo[lodo.Model == model].holdout_f1.mean())
        srows.append({"Model": model, "n_train_full": 6294, "n_train_control": int(r.n_train),
                      "indist_full_mean_3seeds": round(full_mean, 4),
                      "indist_full_seed42": round(float(seed42.macro_f1.iloc[0]), 4) if len(seed42) else np.nan,
                      "size_control_seed42": round(r.macro_f1, 4),
                      "cost_of_less_data_vs_mean": round(full_mean - r.macro_f1, 4),
                      "mean_LODO_holdout_f1": round(lodo_mean, 4),
                      "mean_LODO_drop_vs_full": round(full_mean - lodo_mean, 4)})
    size_tbl = pd.DataFrame(srows)

    logo_tbl.to_csv(os.path.join(OUT, "logo_summary.csv"), index=False)
    size_tbl.to_csv(os.path.join(OUT, "size_control.csv"), index=False)

    lines = ["# Generalisation — leave-one-generator-out and training-size control\n",
             "Normalized Sinhala-HAT, single seed (42), each model's in-distribution LR.\n",
             "## Leave-one-generator-out\n",
             "`indist_same_slice` is the model's in-distribution 3-seed score on the held-out generator's "
             "slice of the test split. `real` marks a drop larger than twice the larger of that slice's "
             "seed spread and the model's overall seed std.\n",
             logo_tbl.to_markdown(index=False), "",
             "\n## Training-size control\n",
             "Same dev/test as the in-distribution runs; training cut to 4,500 rows (the size a held-out "
             "fold trains on). `cost_of_less_data_vs_mean` is the 3-seed in-distribution mean minus the "
             "size-control run; `mean_LODO_drop_vs_full` is the same mean minus the mean leave-one-domain-out "
             "score, for comparison.\n",
             size_tbl.to_markdown(index=False), ""]
    with open(os.path.join(OUT, "GENERALISATION.md"), "w") as f:
        f.write("\n".join(lines))
    return logo_tbl, size_tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse-only", action="store_true", help="rebuild tables from saved runs, no training")
    args = ap.parse_args()

    df = add_pair_generator(load_corpus())
    if not args.analyse_only:
        train_all(df)
    logo_tbl, size_tbl = analyse(df)
    print(logo_tbl.to_string(index=False))
    print(size_tbl.to_string(index=False))
    print("\nDONE ->", os.path.join(OUT, "GENERALISATION.md"))


if __name__ == "__main__":
    main()
