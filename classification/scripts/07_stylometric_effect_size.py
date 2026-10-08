"""
Stylometric effect sizes (Cohen's d): 14 features x 3 domains x 3 generators.

For each of the 14 core stylometric features (src/stat_features.py, CORE_14) and each
domain x generator cell: that generator's AI documents in that domain (Mode A and Mode B
pooled) vs. ALL human documents of that domain. d = (AI mean - human mean) / pooled SD,
so positive d means the feature is higher in AI text. A second set of tables uses the
opposite sign (human - AI), and a pooled row per feature compares all AI with all human
documents.

Reads the raw-text feature table cached by scripts/06_baselines.py.
Outputs in results/stylometric_effect_size/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from data_utils import RESULTS, load_dataset_df
from stat_features import CORE_14

OUT = os.path.join(RESULTS, "stylometric_effect_size")
os.makedirs(OUT, exist_ok=True)
FEATURES = os.path.join(RESULTS, "baselines", "stylometric_features_raw.parquet")


def cohens_d(ai: np.ndarray, human: np.ndarray) -> tuple[float, int, int]:
    n1, n2 = len(ai), len(human)
    v1, v2 = ai.var(ddof=1), human.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    d = (ai.mean() - human.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    return float(d), n1, n2


def main():
    if not os.path.exists(FEATURES):
        raise FileNotFoundError(f"{FEATURES} not found — run scripts/06_baselines.py first")
    feat = pd.read_parquet(FEATURES)
    feat = feat.merge(load_dataset_df()[["text_id", "generator"]], on="text_id", how="left")
    assert feat.generator.notna().all()

    domains = sorted(feat.domain.unique())
    generators = sorted(feat[feat.label == 1].generator.unique())

    rows = []
    for dom in domains:
        human = feat[(feat.domain == dom) & (feat.label == 0)]
        for gen in generators:
            ai = feat[(feat.domain == dom) & (feat.label == 1) & (feat.generator == gen)]
            for f in CORE_14:
                d, n_ai, n_human = cohens_d(ai[f].values, human[f].values)
                rows.append({"domain": dom, "generator": gen, "feature": f,
                             "cohens_d": round(d, 4), "n_ai": n_ai, "n_human": n_human,
                             "ai_mean": round(ai[f].mean(), 4), "human_mean": round(human[f].mean(), 4)})
    table = pd.DataFrame(rows)

    pooled = []
    for f in CORE_14:
        d, n_ai, n_human = cohens_d(feat[feat.label == 1][f].values, feat[feat.label == 0][f].values)
        pooled.append({"feature": f, "cohens_d_ai_minus_human": round(d, 4),
                       "cohens_d_human_minus_ai": round(-d, 4), "n_ai": n_ai, "n_human": n_human})
    pooled = pd.DataFrame(pooled)

    pivot = table.pivot_table(index="feature", columns=["domain", "generator"], values="cohens_d").reindex(CORE_14)
    paper = table.assign(cohens_d=-table.cohens_d)
    paper_pivot = paper.pivot_table(index="feature", columns=["domain", "generator"], values="cohens_d").reindex(CORE_14)

    table.to_csv(os.path.join(OUT, "stylometric_effect_size.csv"), index=False)
    pivot.to_csv(os.path.join(OUT, "stylometric_effect_size_grid.csv"))
    paper.to_csv(os.path.join(OUT, "stylometric_effect_size_human_minus_ai.csv"), index=False)
    paper_pivot.to_csv(os.path.join(OUT, "stylometric_effect_size_grid_human_minus_ai.csv"))
    pooled.to_csv(os.path.join(OUT, "pooled_effect_size.csv"), index=False)

    largest = table.reindex(table.cohens_d.abs().sort_values(ascending=False).index).head(15)
    lines = ["# Stylometric effect sizes (Cohen's d)\n",
             "14 core stylometric features x 3 domains x 3 generators, on Sinhala-HAT as published. "
             "Each cell: that generator's AI documents in that domain (Mode A + Mode B pooled) vs. ALL "
             "human documents of that domain. **d = (AI mean - human mean) / pooled SD — positive d "
             "means the feature is higher in AI text.** The `_human_minus_ai` files use the opposite sign.\n",
             "## Pooled over all domains and generators\n", pooled.to_markdown(index=False), "",
             "\n## Full grid (AI - human)\n", pivot.round(3).to_markdown(), "",
             "\n## 15 largest effects by |d|\n", largest.to_markdown(index=False), ""]
    with open(os.path.join(OUT, "STYLOMETRIC_EFFECT_SIZE.md"), "w") as f:
        f.write("\n".join(lines))
    print(pooled.to_string(index=False))
    print("\nDONE ->", os.path.join(OUT, "STYLOMETRIC_EFFECT_SIZE.md"))


if __name__ == "__main__":
    main()
