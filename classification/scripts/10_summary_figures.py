"""
Summary figures for the README, drawn from the result tables (no training).

  fig_detection.png   test macro-F1 per model with 95% CIs, against the two
                      model-free baselines
  fig_transfer.png    macro-F1 in-distribution vs. on a held-out domain (top row) or
                      held-out generator (bottom row), per model

Reads results/detection, results/baselines and results/generalisation.
Outputs in results/figures/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_utils import MODELS, RESULTS

DET = os.path.join(RESULTS, "detection")
BASE = os.path.join(RESULTS, "baselines")
GEN = os.path.join(RESULTS, "generalisation")
OUT = os.path.join(RESULTS, "figures")
os.makedirs(OUT, exist_ok=True)

# dataviz reference palette: blue ramp steps, neutral ink, light surface
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, BLUE_LIGHT, NEUTRAL = "#2a78d6", "#9ec5f4", "#8a8985"
ORDER = list(MODELS)

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def detection_figure():
    s = pd.read_csv(os.path.join(DET, "indist_summary.csv")).set_index("Model").loc[ORDER]
    s = s.sort_values("test_macro_f1_mean")
    sty = pd.read_csv(os.path.join(BASE, "stylometric_summary.csv"))
    sty = float(sty[(sty.text == "normalized") & (sty.feature_set == "full_15")].test_macro_f1.iloc[0])
    tf = pd.read_csv(os.path.join(BASE, "tfidf_summary.csv"))
    tf = float(tf[(tf.text == "normalized") & (tf.features == "char_2_5")].test_macro_f1.iloc[0])

    rows = [("Stylometric LR (15 features)", sty, None, None, NEUTRAL),
            ("Character 2–5-gram TF-IDF LR", tf, None, None, NEUTRAL)]
    rows += [(m, r.test_macro_f1_mean, r.ci95_low, r.ci95_high, BLUE) for m, r in s.iterrows()]

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    for i, (name, v, lo, hi, color) in enumerate(rows):
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=color, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.plot(v, i, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(max(hi if hi is not None else v, v) + 0.008, i, f"{v:.3f}", va="center", fontsize=10, color=INK)
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(0.502, len(rows) - 0.45, "chance", fontsize=9, color=INK2, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlim(0.48, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Test macro-F1 (675 human–AI pairs)", fontsize=10)
    style(ax)
    fig.suptitle("Fine-tuned encoders beat surface baselines; XLM-R base leads",
                 x=0.012, y=0.97, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.01, "Encoders: mean of 3 seeds, bars are 95% pair-clustered bootstrap intervals. "
             "Baselines: single fit.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.87, bottom=0.17)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_detection.{ext}"), dpi=200)
    plt.close(fig)


def transfer_figure():
    lodo = pd.read_csv(os.path.join(DET, "lodo_summary.csv"))
    logo = pd.read_csv(os.path.join(GEN, "logo_summary.csv")).rename(columns={"held_out_generator": "held_out"})
    labels = {"news": "News", "social_media": "Social media", "wikipedia": "Wikipedia",
              "deepseek-v3": "DeepSeek-V3", "gemini-2.5-pro": "Gemini 2.5 Pro", "gpt-4o": "GPT-4o"}
    panels = [(lodo, ["news", "wikipedia", "social_media"], "domain"),
              (logo, ["deepseek-v3", "gemini-2.5-pro", "gpt-4o"], "generator")]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)
    for r, (tbl, keys, kind) in enumerate(panels):
        for c, key in enumerate(keys):
            ax = axes[r, c]
            t = tbl[tbl.held_out == key].set_index("Model").reindex(ORDER[::-1])
            for i, (m, row) in enumerate(t.iterrows()):
                if pd.isna(row.holdout_f1):
                    continue
                ax.plot([row.holdout_f1, row.indist_same_slice], [i, i], color=GRID, linewidth=3, zorder=1)
                ax.plot(row.indist_same_slice, i, "o", color=BLUE_LIGHT, markersize=9,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
                ax.plot(row.holdout_f1, i, "o", color=BLUE, markersize=9,
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
                ax.text(row.holdout_f1 - 0.015, i, f"{row.holdout_f1:.2f}", ha="right", va="center",
                        fontsize=9, color=INK)
            ax.set_yticks(range(len(ORDER)))
            ax.set_yticklabels(ORDER[::-1] if c == 0 else [], fontsize=10)
            ax.set_ylim(-0.6, len(ORDER) - 0.4)
            ax.set_xlim(0.25, 1.0)
            ax.set_title(f"Held-out {kind}: {labels[key]}", loc="left", fontsize=10.5, color=INK, pad=8)
            style(ax)
            if r == 1:
                ax.set_xlabel("Macro-F1", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=BLUE_LIGHT, markersize=9,
                          label="In-distribution, same test slice"),
               plt.Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=9,
                          label="Held out of training")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.55, 0.0))
    fig.suptitle("Detection fails far more on an unseen domain than on an unseen generator",
                 x=0.012, y=0.985, ha="left", fontsize=12.5, color=INK)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.14, wspace=0.08, hspace=0.35)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_transfer.{ext}"), dpi=200)
    plt.close(fig)


def main():
    detection_figure()
    transfer_figure()
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
