"""
The 14 statistical/stylometric features used by the stylometric logistic-regression
baseline and the effect-size grid, plus the share of Latin tokens (the 15th feature of
the full set). REDUCED drops the four artefact-linked features.

Sinhala has no standard off-the-shelf stemmer available in this environment,
so `stem()` uses a small suffix-stripping heuristic over common Sinhala case/
verb/postposition endings. This is documented as a lightweight heuristic,
not a morphological analyzer, and is used only for the "stem TTR" feature.
"""
from __future__ import annotations

import re

import numpy as np

from normalize import punct_density, count_latin_tokens

# Sentence splitting: Sinhala prose in this corpus is stopped with the
# ordinary Latin full stop / ! / ? (no dedicated Sinhala sentence-final
# punctuation is used in the source domains), so split on those plus
# newlines.
_SENT_SPLIT = re.compile(r"[.!?\n]+")
_WORD_RE = re.compile(r"\S+")

# Heuristic Sinhala suffix list (case markers, common verb/postposition
# endings), longest first so we strip the longest matching suffix.
_SINHALA_SUFFIXES = sorted([
    "වලින්", "වලට", "වලදී", "වෙන්", "ගෙන්", "ලාගේ", "න්ගේ",
    "යින්", "ට්ම", "ෙන්", "ෙහි", "ගේ", "ට", "ෙක්", "යි", "ෙකි",
    "වේ", "යට", "ක්", "ම", "ය", "න්", "ව", "ි",
], key=len, reverse=True)


def stem(word: str) -> str:
    for suf in _SINHALA_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _ttr(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _mattr(words: list[str], window: int = 50) -> float:
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return _ttr(words)
    ratios = []
    for i in range(0, n - window + 1):
        seg = words[i:i + window]
        ratios.append(len(set(seg)) / window)
    return float(np.mean(ratios))


def _ngram_diversity(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def extract_stat_row(text: str, tokenizer=None) -> dict:
    words = _WORD_RE.findall(text)
    n_words = len(words)
    stems = [stem(w) for w in words]
    sents = _sentences(text)
    sent_lens = [len(_WORD_RE.findall(s)) for s in sents] or [n_words]

    ttr = _ttr(words)
    mattr = _mattr(words)
    stem_ttr = _ttr(stems)
    vocab = len(set(words))
    herdan_c = (np.log(vocab) / np.log(n_words)) if n_words > 1 and vocab > 1 else 0.0
    ttr_inflation = (ttr / mattr) if mattr > 0 else 0.0

    if tokenizer is not None and n_words > 0:
        n_subwords = len(tokenizer.tokenize(text))
        fertility = n_subwords / n_words
    else:
        fertility = 0.0

    row = {
        "ttr": ttr,
        "mattr": mattr,
        "stem_ttr": stem_ttr,
        "herdan_c": herdan_c,
        "ttr_inflation": ttr_inflation,
        "bigram_diversity": _ngram_diversity(words, 2),
        "trigram_diversity": _ngram_diversity(words, 3),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sentence_length": float(np.mean(sent_lens)),
        "sentence_length_sd": float(np.std(sent_lens)),
        "punct_density": punct_density(text),
        "fertility": fertility,
        "doc_len_words": n_words,
        "doc_len_chars": len(text),
        # optional artifact-linked feature, added to the "full" set:
        "latin_token_ratio": (count_latin_tokens(text) / n_words) if n_words else 0.0,
    }
    return row


CORE_14 = [
    "ttr", "mattr", "stem_ttr", "herdan_c", "ttr_inflation",
    "bigram_diversity", "trigram_diversity", "avg_word_length",
    "avg_sentence_length", "sentence_length_sd", "punct_density",
    "fertility", "doc_len_words", "doc_len_chars",
]
FULL_15 = CORE_14 + ["latin_token_ratio"]
ARTIFACT_LINKED = ["punct_density", "doc_len_words", "doc_len_chars", "latin_token_ratio"]
REDUCED = [f for f in FULL_15 if f not in ARTIFACT_LINKED]
