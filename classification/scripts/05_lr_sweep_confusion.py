"""
Learning-rate sweep table and confusion-matrix figure for the in-distribution runs.
No training: everything is read from results/detection/ (runs and saved predictions).

  lr_sweep.{csv,md,tex}
      dev macro-F1 by model x learning rate (seed 42, one run per rate); the best rate
      per model (the one the main runs use) is marked.
  confusion_matrix_pooled.{png,pdf}, confusion_counts.csv, confusion_counts_by_domain.csv
      confusion matrices on the test split, all three domains pooled, averaged over the
      3 seeds, all four models. Cell colour and the large number = share of the TRUE
      class; the small number = mean count per seed.

Outputs in results/figures/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from data_utils import MODELS, RESULTS, load_corpus, load_runs, pred_path, splits

DET = os.path.join(RESULTS, "detection")
OUT = os.path.join(RESULTS, "figures")
os.makedirs(OUT, exist_ok=True)

MODEL_ORDER = list(MODELS)
CLASSES = ["Human", "AI"]

# dataviz reference palette: sequential blue (steps 100 -> 700), light surface, ink
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)


# ------------------------------- LR sweep ----------------------------------
def lr_sweep():
    t = load_runs(DET, "search")[["Model", "lr", "dev_macro_f1", "epochs_used"]].copy()
    t["Model"] = pd.Categorical(t.Model, MODEL_ORDER, ordered=True)
    t = t.sort_values(["Model", "lr"]).reset_index(drop=True)
    t["best"] = t.groupby("Model", observed=True).dev_macro_f1.transform("max") == t.dev_macro_f1
    t["dev_macro_f1"] = t.dev_macro_f1.round(4)
    t.to_csv(os.path.join(OUT, "lr_sweep.csv"), index=False)

    md = ["# Learning-rate sweep\n",
          "Dev macro-F1, one run per rate (seed 42, up to 12 epochs, early stopping patience 2, batch 16). "
          "The best rate per model is what the main runs use.\n",
          t.to_markdown(index=False), ""]
    open(os.path.join(OUT, "lr_sweep.md"), "w").write("\n".join(md))

    tex = ["\\begin{tabular}{lrc}", "\\toprule", "Model & LR & Dev macro-F1 \\\\", "\\midrule"]
    last = None
    for _, r in t.iterrows():
        name = str(r.Model) if r.Model != last else ""
        if last is not None and r.Model != last:
            tex.append("\\midrule")
        v = f"{r.dev_macro_f1:.4f}"
        cell = f"\\textbf{{{v}}}" if r.best else v
        tex.append(f"{name} & ${r.lr * 1e5:.0f}\\times10^{{-5}}$ & {cell} \\\\")
        last = r.Model
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "lr_sweep.tex"), "w").write("\n".join(tex) + "\n")
    return t


# ---------------------------- confusion matrices -----------------------------
def luminance(hex_or_rgb):
    rgb = matplotlib.colors.to_rgb(hex_or_rgb)
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def text_on(bg):
    return "#ffffff" if contrast(bg, "#ffffff") >= contrast(bg, INK) else INK


def confusion(y, p):
    m = np.zeros((2, 2), dtype=float)
    for t in (0, 1):
        for q in (0, 1):
            m[t, q] = ((y == t) & (p == q)).sum()
    return m


def confusion_figure():
    te = splits(load_corpus())[2]
    y = te.label.values
    main = load_runs(DET, "main")
    summ = pd.read_csv(os.path.join(DET, "indist_summary.csv")).set_index("Model")

    mats, dom_rows, cnt_rows = {}, [], []
    for model in MODEL_ORDER:
        runs = main[main.Model == model].sort_values("seed")
        preds = [np.load(pred_path(DET, r)) for r in runs.run_id]
        assert len(preds) == 3 and all(len(p) == len(y) for p in preds)
        mats[model] = np.mean([confusion(y, p) for p in preds], axis=0)
        m = mats[model]
        cnt_rows.append({"Model": model, "n_seeds": len(preds), "true_human_pred_human": m[0, 0],
                         "true_human_pred_AI": m[0, 1], "true_AI_pred_human": m[1, 0], "true_AI_pred_AI": m[1, 1]})
        for dom in sorted(te.domain.unique()):
            mk = (te.domain == dom).values
            md = np.mean([confusion(y[mk], p[mk]) for p in preds], axis=0)
            dom_rows.append({"Model": model, "domain": dom, "true_human_pred_human": md[0, 0],
                             "true_human_pred_AI": md[0, 1], "true_AI_pred_human": md[1, 0],
                             "true_AI_pred_AI": md[1, 1]})
    pd.DataFrame(cnt_rows).round(2).to_csv(os.path.join(OUT, "confusion_counts.csv"), index=False)
    pd.DataFrame(dom_rows).round(2).to_csv(os.path.join(OUT, "confusion_counts_by_domain.csv"), index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.7), facecolor=SURFACE)
    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        m = mats[model]
        rate = m / m.sum(axis=1, keepdims=True)
        ax.set_facecolor(SURFACE)
        ax.pcolormesh(np.arange(3), np.arange(3), rate, cmap=CMAP, vmin=0, vmax=1,
                      edgecolors=SURFACE, linewidth=3)
        ax.set_xlim(0, 2); ax.set_ylim(2, 0)
        ax.set_aspect("equal")
        for r in range(2):
            for c in range(2):
                bg = CMAP(rate[r, c])
                fg = text_on(bg)
                ax.text(c + 0.5, r + 0.42, f"{rate[r, c]:.1%}", ha="center", va="center",
                        fontsize=15, fontweight="bold", color=fg)
                ax.text(c + 0.5, r + 0.68, f"{m[r, c]:.0f} docs", ha="center", va="center",
                        fontsize=8.5, color=fg)
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(CLASSES, fontsize=10)
        ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(CLASSES if i == 0 else ["", ""], fontsize=10)
        ax.tick_params(length=0)
        ax.set_xlabel("Predicted", fontsize=10)
        if i == 0:
            ax.set_ylabel("True class", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        f1 = summ.loc[model, "test_macro_f1_mean"]
        ax.set_title(f"{model}\nmacro-F1 {f1:.3f}", fontsize=11, color=INK, pad=8, loc="left")
    fig.subplots_adjust(left=0.06, right=0.90, top=0.80, bottom=0.14, wspace=0.10)
    cax = fig.add_axes([0.925, 0.22, 0.014, 0.50])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Share of true class", fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    cb.outline.set_visible(False)
    fig.suptitle("Confusion matrices, all three domains pooled "
                 "(mean of 3 seeds, test split, 675 human + 675 AI documents)",
                 x=0.06, y=0.985, ha="left", fontsize=11, color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"confusion_matrix_pooled.{ext}"), dpi=220, facecolor=SURFACE)
    plt.close(fig)
    return mats


def main():
    t = lr_sweep()
    print(t.to_string(index=False))
    mats = confusion_figure()
    for k, v in mats.items():
        print(k, v.round(1).tolist())
    print("\nDONE ->", OUT)


if __name__ == "__main__":
    main()
