#!/usr/bin/env python3
"""Source-data quality filter for the Wikipedia domain, run before allocation.

Motivated by two failure modes seen in pilot generation (logs/logs/pilot_results_5.jsonl,
source_id 8633 and 13740): a category-tag-only "article" (human_text was just three
``ප්‍රවර්ගය:`` lines, no prose) and a disambiguation page (human_text was a "X can refer
to:" list) were both sampled as generation sources. With no real content to draw on,
every model fabricated a full invented article from the title alone.

Unlike filtering.py's filter_source_items() (which runs on the plain list[dict] just
before allocate_source_items() shuffles it), this runs on the pandas DataFrame right
after load_source_data(), so the per-check columns can be inspected in bulk before the
DataFrame is ever converted to a list of records.
"""

import pandas as pd

CATEGORY_PREFIX = "ප්‍රවර්ගය:"

# "යොමු විය හැක්කේ" ("can be referred to") and "වෙත යොමු කරයි" ("redirects to/leads to")
# together hit 30 rows across the full 26,470-row dataset. "විය හැකි" ("could be/might
# be") was deliberately left out: it's an ordinary Sinhala modal phrase that shows up in
# 10.5% of all rows (2,773/26,470) as part of normal sentences (e.g. "...ටර්මිනල වෙත
# යළි යැවිය හැකි..."), so including it would reject roughly 1 in 10 legitimate articles.
DISAMBIGUATION_MARKERS = ("යොමු විය හැක්කේ", "වෙත යොමු කරයි")

# A real disambiguation page is a short list, not an article: of the 30 rows hit by the
# markers above, genuine disambiguation pages topped out at 89 words and the next one up
# jumped straight to 213 (some as long as 14,299 — e.g. "ඊජිප්තුව"/Egypt). In those, the
# marker is just an ordinary verb phrase inside real prose — e.g. "තණ්හාව" ("craving")
# uses "වෙත යොමු කරයි" to mean craving "leads to" defilements, nothing to do with
# disambiguation. 100 sits in that clean gap.
DISAMBIGUATION_MAX_WORDS = 100

RESIDUAL_MARKUP_TOKENS = ("thumb|", "right|", "left|", "link=", "px|", "[[File:", "[[Image:")

SINHALA_RANGE = (0x0D80, 0x0DFF)
ASCII_MAX = 0x7F


def is_category_only(text: str) -> bool:
    """True if every non-empty line starts with the category prefix — i.e. the
    "article" is just category tags with no real prose."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return bool(lines) and all(line.startswith(CATEGORY_PREFIX) for line in lines)


def is_disambiguation(text: str, max_words: int = DISAMBIGUATION_MAX_WORDS) -> bool:
    """True if the text contains a disambiguation marker AND is short enough to
    actually be a disambiguation list rather than a real article that happens to use
    the marker phrase in ordinary prose (see DISAMBIGUATION_MAX_WORDS)."""
    text = text or ""
    if not any(marker in text for marker in DISAMBIGUATION_MARKERS):
        return False
    return word_count(text) < max_words


def word_count(text: str) -> int:
    return len((text or "").split())


def has_residual_markup(text: str) -> bool:
    text = text or ""
    return any(token in text for token in RESIDUAL_MARKUP_TOKENS)


def has_encoding_corruption(text: str, threshold: float = 0.15) -> bool:
    """True if more than `threshold` of the characters fall outside the Sinhala
    Unicode block and basic ASCII — catches legacy-font mojibake/garbled encoding.

    Deliberately treats plain ASCII (including English words) as NOT corrupted —
    a well-formed English article isn't mojibake. Wrong-language content is a
    separate concern, see has_insufficient_sinhala()."""
    text = text or ""
    if not text:
        return False
    outside = sum(
        1 for ch in text
        if not (SINHALA_RANGE[0] <= ord(ch) <= SINHALA_RANGE[1] or ord(ch) <= ASCII_MAX)
    )
    return (outside / len(text)) > threshold


def has_insufficient_sinhala(text: str, min_ratio: float = 0.5) -> bool:
    """True if fewer than `min_ratio` of the alphabetic characters are Sinhala-script.

    Catches leftover untranslated stub articles copied verbatim from English Wikipedia
    (e.g. source_id 5731, title "ශින්ටෝ", body starting "Shinto (神道, Shintō?) is the
    native religion of Japan..." — a Sinhala title over a fully English body). Only
    alphabetic characters are counted (digits/punctuation ignored) so a normal article
    with a parenthetical foreign name, e.g. "අයිවර් ඩෙනිස් (Ivor Dennis)", isn't
    penalized. Across the full dataset, legitimate articles cluster at a 0.87-1.0
    Sinhala ratio (25th percentile 0.87); a 0.5 floor cleanly separates those from
    genuinely wrong-language rows without rejecting normal name/term mixing."""
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return False
    sinhala = sum(1 for ch in letters if SINHALA_RANGE[0] <= ord(ch) <= SINHALA_RANGE[1])
    return (sinhala / len(letters)) < min_ratio


def quality_check(row: dict, min_words: int = 100) -> dict:
    """Run every check against one source row's human_text (and title, for the
    encoding-corruption check, since a garbled title can accompany a clean body or
    vice versa). Returns {check_name: passed} plus an overall "passed" boolean."""
    text = row.get("human_text") or ""
    title = row.get("title") or ""

    checks = {
        "not_category_only": not is_category_only(text),
        "not_disambiguation": not is_disambiguation(text),
        "sufficient_length": word_count(text) >= min_words,
        "no_residual_markup": not has_residual_markup(text),
        "no_encoding_corruption": not (has_encoding_corruption(text) or has_encoding_corruption(title)),
        "sufficient_sinhala_content": not has_insufficient_sinhala(text),
    }
    checks["passed"] = all(checks.values())
    return checks


def filter_source_data(df: pd.DataFrame, min_words: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply quality_check to every row. Returns (clean, rejected): clean keeps only
    passing rows with the check columns dropped; rejected keeps every check column
    (including which ones failed) for manual review."""
    check_df = pd.DataFrame(
        [quality_check(row, min_words=min_words) for row in df.to_dict("records")],
        index=df.index,
    )
    combined = pd.concat([df, check_df], axis=1)
    passed_mask = combined["passed"]

    clean = combined[passed_mask].drop(columns=list(check_df.columns)).reset_index(drop=True)
    rejected = combined[~passed_mask].reset_index(drop=True)
    return clean, rejected
