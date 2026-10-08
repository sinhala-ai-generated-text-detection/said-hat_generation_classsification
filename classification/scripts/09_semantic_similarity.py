"""
Semantic preservation via embedding similarity.

For every AI document paired with a human source, computes:
  - LaBSE cosine similarity (mean-pooled, L2-normalized embeddings)
  - Multilingual BERTScore F1 (model_type=xlm-roberta-base, no baseline
    rescaling — no Sinhala baseline exists, so scores are raw, not rescaled)

Two groups:
  mode_b — the paraphrase, compared against the human source it paraphrases.
           This is the main result: high similarity is the hypothesis.
  mode_a — the title-conditioned document, compared against the SAME paired
           human document even though it was never shown that document's
           body, only its title. This is the deliberate LOWER-SIMILARITY
           reference point, not a paraphrase-quality
           result in its own right.

Inference only (no fine-tuning). Both texts are truncated to the first 500
whitespace words before embedding/scoring, since a few human sources run to
several thousand words (max 12,120) — this keeps compute and memory bounded
and is applied identically to both sides and both groups.

Named-entity retention is NOT computed here (no Sinhala NER tool identified).

--summarise-only rebuilds the summary tables and report from the saved per-pair scores
without loading any model.

Outputs to results/dataset_eval/:
  semantic_pairs_mode_{a,b}.csv   one row per pair
  semantic_summary_by_domain.csv, semantic_summary_by_generator.csv
  SEMANTIC_SIMILARITY.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from data_utils import RESULTS, load_dataset_df

OUT = os.path.join(RESULTS, "dataset_eval")
os.makedirs(OUT, exist_ok=True)
LABSE_MODEL = "sentence-transformers/LaBSE"
BERTSCORE_MODEL = "xlm-roberta-base"
MAX_WORDS = 500
BATCH_SIZE = 32


def truncate(text: str, n_words: int = MAX_WORDS) -> str:
    return " ".join(text.split()[:n_words])


def build_pairs(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    ai = df[df.generation_mode == mode].copy()
    human = df[df.label == 0].set_index("group")
    assert set(ai.group) <= set(human.index)
    ai["src_text"] = human.loc[ai.group, "text"].values
    ai["ai_text_trunc"] = ai.text.map(truncate)
    ai["src_text_trunc"] = ai.src_text.map(truncate)
    return ai[["text_id", "group", "domain", "generator", "generation_mode",
               "ai_text_trunc", "src_text_trunc"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# LaBSE cosine similarity
# --------------------------------------------------------------------------
def labse_embed(texts: list[str], tok, model, device: str) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy())
            print(f"    LaBSE embed {min(i + BATCH_SIZE, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(out, axis=0)


def labse_cosine(pairs: pd.DataFrame, tok, model, device: str) -> np.ndarray:
    a = labse_embed(pairs.ai_text_trunc.tolist(), tok, model, device)
    b = labse_embed(pairs.src_text_trunc.tolist(), tok, model, device)
    return (a * b).sum(axis=1)


# --------------------------------------------------------------------------
# BERTScore
# --------------------------------------------------------------------------
def bertscore_f1(pairs: pd.DataFrame, device: str) -> np.ndarray:
    # bert_score's mask outer-product hits a Triton kernel; needs
    # python3.12-dev (Python.h) installed for gcc to JIT-compile it. Now
    # installed, so this runs on GPU like everything else.
    from bert_score import score as bert_score_fn
    _, _, f1 = bert_score_fn(pairs.ai_text_trunc.tolist(), pairs.src_text_trunc.tolist(),
                             model_type=BERTSCORE_MODEL, batch_size=BATCH_SIZE,
                             device=device, verbose=True, rescale_with_baseline=False)
    return f1.numpy()


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg_kwargs = dict(n=("labse_cosine", "size"),
                       labse_cosine_mean=("labse_cosine", "mean"),
                       labse_cosine_std=("labse_cosine", "std"),
                       bertscore_f1_mean=("bertscore_f1", "mean"),
                       bertscore_f1_std=("bertscore_f1", "std"))
    if by:
        return df.groupby(by).agg(**agg_kwargs).round(4)
    return pd.DataFrame([{k: getattr(df[v[0]], v[1])() if v[1] != "size" else len(df)
                          for k, v in agg_kwargs.items()}]).round(4)


def compute():
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    df = load_dataset_df()
    assert len(df) == 8994

    mb = build_pairs(df, "mode_b")
    ma = build_pairs(df, "mode_a")
    print(f"mode_b pairs: {len(mb)}  mode_a pairs: {len(ma)}")

    print("\nloading LaBSE...", flush=True)
    labse_tok = AutoTokenizer.from_pretrained(LABSE_MODEL)
    labse_model = AutoModel.from_pretrained(LABSE_MODEL).to(device).eval()

    for name, pairs in [("mode_b", mb), ("mode_a", ma)]:
        print(f"\n--- LaBSE cosine: {name} ({len(pairs)} pairs) ---", flush=True)
        pairs["labse_cosine"] = labse_cosine(pairs, labse_tok, labse_model, device)

    del labse_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nloading BERTScore model...", flush=True)
    for name, pairs in [("mode_b", mb), ("mode_a", ma)]:
        print(f"\n--- BERTScore: {name} ({len(pairs)} pairs) ---", flush=True)
        pairs["bertscore_f1"] = bertscore_f1(pairs, device)

    mb = mb.drop(columns=["ai_text_trunc", "src_text_trunc"])
    ma = ma.drop(columns=["ai_text_trunc", "src_text_trunc"])
    mb.to_csv(os.path.join(OUT, "semantic_pairs_mode_b.csv"), index=False)
    ma.to_csv(os.path.join(OUT, "semantic_pairs_mode_a.csv"), index=False)
    return mb, ma



def write_report(mb, ma):
    by_domain_b = summarize(mb, ["domain"])
    by_gen_b = summarize(mb, ["generator"])
    by_dom_gen_b = summarize(mb, ["domain", "generator"])
    overall_b = summarize(mb, []); overall_b.index = ["overall"]
    by_domain_a = summarize(ma, ["domain"])
    overall_a = summarize(ma, []); overall_a.index = ["overall"]

    for name, tbl in [("mode_b_by_domain", by_domain_b), ("mode_b_by_generator", by_gen_b),
                      ("mode_b_by_domain_generator", by_dom_gen_b), ("mode_b_overall", overall_b),
                      ("mode_a_by_domain", by_domain_a), ("mode_a_overall", overall_a)]:
        tbl.to_csv(os.path.join(OUT, f"semantic_summary_{name}.csv"))

    lines = ["# Semantic preservation via embedding similarity\n",
             f"LaBSE cosine similarity and multilingual BERTScore F1 ({BERTSCORE_MODEL}, no baseline "
             f"rescaling), between each AI document and its paired human source. Both texts truncated "
             f"to the first {MAX_WORDS} words before scoring. Inference only, no fine-tuning.\n",
             "## Mode B (paraphrase vs. its source) — the main result\n",
             "### Overall\n", overall_b.to_markdown(), "",
             "\n### By domain\n", by_domain_b.to_markdown(), "",
             "\n### By generator\n", by_gen_b.to_markdown(), "",
             "\n### By domain x generator\n", by_dom_gen_b.to_markdown(), "",
             "\n## Mode A (title-conditioned) vs. the SAME paired human document — lower-similarity reference\n",
             "Mode A never saw this document, only its title. Low similarity here is expected and is "
             "what makes the Mode B numbers above meaningful, not a paraphrase-quality result on its own.\n",
             "### Overall\n", overall_a.to_markdown(), "",
             "\n### By domain\n", by_domain_a.to_markdown(), "",
             "\n## Note\n",
             "Named-entity retention was not computed here (no Sinhala NER tool identified).\n"]
    with open(os.path.join(OUT, "SEMANTIC_SIMILARITY.md"), "w") as f:
        f.write("\n".join(lines))

    print("\n=== Mode B (main result) ===")
    print(overall_b.to_string())
    print(by_domain_b.to_string())
    print("\n=== Mode A (reference) ===")
    print(overall_a.to_string())
    print("\nDONE ->", os.path.join(OUT, "SEMANTIC_SIMILARITY.md"))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild the summary tables from the saved per-pair scores, no models")
    if ap.parse_args().summarise_only:
        mb = pd.read_csv(os.path.join(OUT, "semantic_pairs_mode_b.csv"))
        ma = pd.read_csv(os.path.join(OUT, "semantic_pairs_mode_a.csv"))
    else:
        mb, ma = compute()
    write_report(mb, ma)


if __name__ == "__main__":
    main()
