"""Shared data loading / integrity-check helpers."""
from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
CORPUS_PATH = os.path.join(RESULTS, "corpus", "said_hat_normalized.parquet")

HF_DATASET = "said-sinhala-ai-dataset/said-hat"

MODELS = {
    "XLM-R base": "xlm-roberta-base",
    "SinhalaBERTo": "keshan/SinhalaBERTo",
    "LaBSE": "sentence-transformers/LaBSE",
    "SinBERT-large": "NLPC-UOM/SinBERT-large",
}


def load_dataset_df(config: str = "default") -> pd.DataFrame:
    """Sinhala-HAT from the Hugging Face Hub, all splits in one DataFrame. The Hub's
    `validation` split is renamed `dev`, the name every script uses.

    Rows are put in a fixed order (split, then label, then domain, keeping the Hub's
    order within each block). Saved predictions (preds_*.npy) are aligned to this order,
    so it must not change."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, config)
    df = pd.concat([ds[s].to_pandas() for s in ("train", "validation", "test")], ignore_index=True)
    df["split"] = df["split"].replace({"validation": "dev"})
    split_rank = df.split.map({"train": 0, "dev": 1, "test": 2})
    order = pd.DataFrame({"s": split_rank, "l": df.label, "d": df.domain}).sort_values(
        ["s", "l", "d"], kind="mergesort").index
    df = df.loc[order].reset_index(drop=True)
    if config == "default":
        verify_pairing_and_splits(df)
    return df


def load_corpus() -> pd.DataFrame:
    """The normalized training corpus built by scripts/01_build_corpus.py."""
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"{CORPUS_PATH} not found — run scripts/01_build_corpus.py first")
    df = pd.read_parquet(CORPUS_PATH)
    verify_pairing_and_splits(df)
    return df


def add_pair_generator(df: pd.DataFrame) -> pd.DataFrame:
    """Tag both rows of each pair with the generator of its AI member, so a pair can be
    held out by generator."""
    pg = df[df.label == 1].set_index("group").generator.to_dict()
    df = df.copy()
    df["pair_generator"] = df.group.map(pg)
    assert df.pair_generator.notna().all()
    return df


def verify_pairing_and_splits(df: pd.DataFrame) -> None:
    """Every pair has one human and one AI row, no pair crosses splits, splits are the
    published sizes and 50/50 label balanced."""
    assert len(df) == 8994, f"expected 8994 rows, got {len(df)}"

    gsizes = df.groupby("group").size()
    assert (gsizes == 2).all(), "not every group has exactly one human + one AI row"

    train_groups = set(df[df.split == "train"].group)
    dev_groups = set(df[df.split == "dev"].group)
    test_groups = set(df[df.split == "test"].group)
    leak = (train_groups | dev_groups) & test_groups
    assert not leak, f"LEAK — {len(leak)} groups shared with test"
    leak2 = train_groups & dev_groups
    assert not leak2, f"LEAK — {len(leak2)} groups shared between train and dev"

    counts = df.groupby("split").size()
    assert counts.get("train", 0) == 6294
    assert counts.get("dev", 0) == 1350
    assert counts.get("test", 0) == 1350

    for split_name, grp in df.groupby("split"):
        vc = grp.label.value_counts()
        assert vc.get(0, 0) == vc.get(1, 0), f"{split_name} split is not 50/50 label balanced"

    dom_label = df.groupby(["domain", "label"]).size()
    assert dom_label.min() > 0, "a domain/label combination is empty"


def splits(df: pd.DataFrame):
    return tuple(df[df.split == s].reset_index(drop=True) for s in ("train", "dev", "test"))


def holdout_split(seen: pd.DataFrame, dev_frac: float = 0.25, seed: int = 42):
    """75/25 train/dev re-split of the seen rows for a held-out fold, grouped by pair."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=dev_frac, random_state=seed)
    a, b = next(gss.split(seen, groups=seen.group))
    tr, dv = seen.iloc[a].reset_index(drop=True), seen.iloc[b].reset_index(drop=True)
    assert not (set(tr.group) & set(dv.group)), "pair split across train and dev"
    return tr, dv


def pred_path(results_dir: str, run_id: str) -> str:
    """Where train.Engine saves a run's test predictions."""
    return os.path.join(results_dir, f"preds_{run_id.replace('|', '_').replace('/', '-')}.npy")


def load_runs(results_dir: str, phase: str | None = None) -> pd.DataFrame:
    r = pd.read_csv(os.path.join(results_dir, "runs.csv")).drop_duplicates("run_id", keep="last")
    return r[r.phase == phase] if phase else r
