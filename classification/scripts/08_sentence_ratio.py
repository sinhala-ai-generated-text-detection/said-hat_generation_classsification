"""
Sentence-count ratio between each paraphrase and its human source (discourse structure).

For every Mode B (paraphrase) document, count sentences on both the AI text
and its paired human source, then take the ratio ai_sentences / source_sentences
— the same direction as the already-reported word-count ratio ("the length
ratio between each paraphrase and its human source"). Sentence counting uses
the deterministic Sinhala segmenter (src/sinhala_segment.py).

Mode A (title-conditioned) is not a paraphrase of its paired human document, so it is
excluded from the main result and reported separately for context.

No model inference: pure counting over the Hub dataset (un-normalized text).

Outputs to results/dataset_eval/:
  sentence_ratio_pairs.csv   one row per pair
  sentence_ratio_summary.csv mean/median/std per domain, generator, overall
  SENTENCE_RATIO.md
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from data_utils import RESULTS, load_dataset_df
from sinhala_segment import sentence_count

OUT = os.path.join(RESULTS, "dataset_eval")
os.makedirs(OUT, exist_ok=True)


def build_pairs(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    ai = df[df.generation_mode == mode].copy()
    human = df[df.label == 0].set_index("group")
    assert set(ai.group) <= set(human.index), f"{mode}: some pairs have no human source"

    ai["src_text"] = human.loc[ai.group, "text"].values
    ai["src_word_count"] = human.loc[ai.group, "word_count"].values

    ai["ai_sentences"] = ai.text.map(sentence_count)
    ai["src_sentences"] = ai.src_text.map(sentence_count)
    ai["sentence_ratio"] = ai.ai_sentences / ai.src_sentences.replace(0, np.nan)
    ai["word_ratio"] = ai.word_count / ai.src_word_count.replace(0, np.nan)
    return ai[["text_id", "group", "domain", "generator", "generation_mode",
               "word_count", "src_word_count", "word_ratio",
               "ai_sentences", "src_sentences", "sentence_ratio"]]


def summarize(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg_kwargs = dict(n=("sentence_ratio", "size"),
                       sentence_ratio_mean=("sentence_ratio", "mean"),
                       sentence_ratio_median=("sentence_ratio", "median"),
                       sentence_ratio_std=("sentence_ratio", "std"),
                       word_ratio_mean=("word_ratio", "mean"),
                       word_ratio_median=("word_ratio", "median"))
    if by:
        return pairs.groupby(by).agg(**agg_kwargs).round(4)
    return pd.DataFrame([{k: getattr(pairs[v[0]], v[1])() if v[1] != "size" else len(pairs)
                          for k, v in agg_kwargs.items()}]).round(4)


def main():
    df = load_dataset_df()

    mb = build_pairs(df, "mode_b")
    ma = build_pairs(df, "mode_a")
    mb.to_csv(os.path.join(OUT, "sentence_ratio_pairs_mode_b.csv"), index=False)
    ma.to_csv(os.path.join(OUT, "sentence_ratio_pairs_mode_a.csv"), index=False)

    by_domain = summarize(mb, ["domain"])
    by_gen = summarize(mb, ["generator"])
    by_dom_gen = summarize(mb, ["domain", "generator"])
    overall = summarize(mb, [])
    overall.index = ["overall"]
    ma_by_domain = summarize(ma, ["domain"])
    ma_overall = summarize(ma, [])
    ma_overall.index = ["overall (mode A)"]

    for name, tbl in [("by_domain", by_domain), ("by_generator", by_gen),
                      ("by_domain_generator", by_dom_gen), ("overall", overall),
                      ("mode_a_by_domain", ma_by_domain), ("mode_a_overall", ma_overall)]:
        tbl.to_csv(os.path.join(OUT, f"sentence_ratio_{name}.csv"))

    lines = ["# Sentence-count ratio\n",
             f"Mode B (paraphrase) pairs: n={len(mb)}. Ratio = AI sentence count / human-source "
             "sentence count, sentences counted with the deterministic Sinhala segmenter "
             "(abbreviation/decimal/initial/quote-aware). Word-count ratio is included alongside "
             "as a cross-check against the already-reported 0.89-0.99 figure.\n",
             "## Overall (Mode B)\n", overall.to_markdown(), "",
             "\n## By domain (Mode B)\n", by_domain.to_markdown(), "",
             "\n## By generator (Mode B)\n", by_gen.to_markdown(), "",
             "\n## By domain x generator (Mode B)\n", by_dom_gen.to_markdown(), "",
             "\n## For context: Mode A (title-conditioned), vs. the same paired human document\n",
             "Mode A does not paraphrase this document — it was written from only the title. "
             "Included as a reference point only.\n",
             ma_overall.to_markdown(), "", ma_by_domain.to_markdown(), ""]
    with open(os.path.join(OUT, "SENTENCE_RATIO.md"), "w") as f:
        f.write("\n".join(lines))

    print(overall.to_string())
    print()
    print(by_domain.to_string())
    print("\nDONE ->", os.path.join(OUT, "SENTENCE_RATIO.md"))


if __name__ == "__main__":
    main()
