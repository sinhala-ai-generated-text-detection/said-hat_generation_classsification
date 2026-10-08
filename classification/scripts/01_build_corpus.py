"""
Build the training corpus from Sinhala-HAT.

Downloads the dataset from the Hugging Face Hub and applies the same normalization to
both classes (src/normalize.py): Latin-script tokens -> <LAT>, emoji and residual wiki
markup removed, punctuation variants and whitespace collapsed. Word counts are
recomputed after normalization.

Output: results/corpus/said_hat_normalized.parquet, read by every training script.
"""
import os
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_utils import CORPUS_PATH, load_dataset_df, verify_pairing_and_splits
from normalize import normalize_text


def has_punctuation(text: str) -> bool:
    return any(unicodedata.category(ch)[0] in "PS" for ch in text.replace("<LAT>", ""))


def main():
    df = load_dataset_df()
    print(f"loaded {len(df)} documents from the Hub, integrity ok")

    sm = df.domain == "social_media"
    assert not df.loc[sm, "text"].map(has_punctuation).any(), "social-media text should have no punctuation"

    out = df.copy()
    out["text"] = out["text"].map(normalize_text)
    out["word_count"] = out["text"].str.split().map(len)
    verify_pairing_and_splits(out)
    assert not out.loc[sm, "text"].map(has_punctuation).any(), "normalization added punctuation to social media"

    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    out.to_parquet(CORPUS_PATH)
    print(f"saved {CORPUS_PATH} ({len(out)} rows, mean word count "
          f"{df.word_count.mean():.1f} -> {out.word_count.mean():.1f})")


if __name__ == "__main__":
    main()
