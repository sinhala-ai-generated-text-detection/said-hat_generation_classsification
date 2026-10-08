#!/usr/bin/env python3
"""Source-data quality filter for the Wikipedia domain, run before allocation.

Motivated by two failure modes seen in pilot generation (logs/logs/pilot_results_5.jsonl,
source_id 8633 and 13740): a category-tag-only "article" (human_text was just three
``ප්‍රවර්ගය:`` lines, no prose) and a disambiguation page (human_text was a "X can refer
to:" list) were both sampled as generation sources. With no real content to draw on,
every model fabricated a full invented article from the title alone.

Unlike filtering.py's filter_source_items() (which runs on the plain list[dict] just
before allocation.py's slice_model_items() shuffles it), this runs on the pandas
DataFrame right after load_source_data(), so the per-check columns can be inspected in
bulk before the DataFrame is ever converted to a list of records.
"""

import re
import unicodedata

import pandas as pd

CATEGORY_PREFIX = "ප්‍රවර්ගය:"

# The wikitext -> text conversion leaves the article's category tags behind as trailing
# lines ("ප්‍රවර්ගය:මානව සම්පත් කළමනාකරණය", or English "Category:X"): 634 of the 1,500
# sampled human texts (42%) carried one, versus 21 of the 1,500 AI outputs — the AI
# paraphrase drops them and title-only generation never sees them — so their presence
# alone would tell a classifier "human". The prefix appears both with and without the
# zero-width joiner (U+200D) after the virama, hence the optional ‍.
CATEGORY_LINE_RE = re.compile(
    # optional leading "[[" so a raw wikitext leftover like "[[Category:X]]" is caught too
    r"^[ \t]*(?:\[\[)?[ \t]*(?:ප්‍?රවර්ගය|Category)[ \t]*:.*$",
    re.MULTILINE,
)


# A tag can also be glued onto the end of a sentence with no line break ("...උත්සහ දරයි.ප්‍රවර්ගය:X"):
# a tag marker immediately after sentence-ending punctuation, through the end of that line. Requiring the
# preceding sentence end keeps ordinary uses of the word ("award category", "ප්‍රවර්ගයට") untouched.
INLINE_CATEGORY_TAG_RE = re.compile(
    r"(?<=[.!?\u0964])[ \t]*(?:\[\[)?[ \t]*(?:\u0db4\u0dca\u200d?\u0dbb\u0dc0\u0dbb\u0dca\u0d9c\u0dba|Category)[ \t]*:[^\n]*$",
    re.MULTILINE,
)


def strip_category_tags(text: str) -> str:
    """Remove every category tag — whole tag lines, and tags glued onto the end of a sentence — plus the
    blank lines left behind. Text with no tag is returned untouched (byte-identical), so only affected
    rows change."""
    text = text or ""
    if not (CATEGORY_LINE_RE.search(text) or INLINE_CATEGORY_TAG_RE.search(text)):
        return text
    text = CATEGORY_LINE_RE.sub("", INLINE_CATEGORY_TAG_RE.sub("", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()

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

# ChatGPT's public launch (2022-11-30) — the point mass LLM-assisted editing of Wikipedia
# became plausible for ordinary editors, vs. GPT-3's mid-2020 API release, which stayed
# paid/limited and never saw casual use on si.wikipedia.org. Requires created_date AND
# last_modified_date (added by build_wikipedia_dated_dataset.py, which fetches both from
# the MediaWiki API since the source parquet itself carries no timestamps) to fall before
# this cutoff, so an article that was created early but rewritten/expanded after the
# cutoff is excluded too, not just one created late.
GPT_ERA_CUTOFF = pd.Timestamp("2022-11-30", tz="UTC")


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


def _counts_as_foreign_script(ch: str) -> bool:
    """True for a character that is evidence of mojibake/garbled encoding: a letter or mark from a
    script other than Sinhala/ASCII (Cyrillic, Latin-1 accents, ...), or a junk code point that
    legacy-font corruption tends to land on — unassigned, private-use, surrogate, stray control
    characters, and the U+FFFD replacement character. Everything else is ignored:
      * format characters (zero-width joiner/non-joiner U+200C/D, ZWSP, BOM) — Sinhala conjuncts
        such as ක්‍ර, ප්‍ර are *written* with a ZWJ, so Sinhala text is full of them;
      * punctuation, symbols and emoji, and spaces/separators (curly quotes, dashes, ©, 😀, ═══);
      * emoji plumbing that is technically a Mark: variation selectors (U+FE00-FE0F) and the
        symbol-combining marks (U+20D0-20FF, e.g. the keycap in 1️⃣)."""
    o = ord(ch)
    if o <= ASCII_MAX or SINHALA_RANGE[0] <= o <= SINHALA_RANGE[1]:
        return False
    if o == 0xFFFD:
        return True
    cat = unicodedata.category(ch)
    if cat[0] in ("L", "M"):
        return not (0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF or 0x20D0 <= o <= 0x20FF)
    return cat in ("Cn", "Co", "Cs", "Cc")


def has_encoding_corruption(text: str, threshold: float = 0.15) -> bool:
    """True if more than `threshold` of the visible characters are letters/marks from a script
    other than Sinhala or ASCII — catches legacy-font mojibake/garbled encoding.

    Deliberately treats plain ASCII (including English words) as NOT corrupted —
    a well-formed English article isn't mojibake. Wrong-language content is a
    separate concern, see has_insufficient_sinhala().

    Only letters and marks count (see _counts_as_foreign_script): the previous version counted
    *any* character outside the Sinhala block and ASCII, so Sinhala text heavy in zero-width
    joiners, curly quotes, or an emoji-heavy social-media caption could be rejected as
    "corrupted" despite being perfectly well-formed. Real mojibake still trips it because the
    garbage is mostly letters (Cyrillic/accented Latin) — e.g. UTF-8 Sinhala misread as
    Windows-1251 or Latin-1. Invisible format characters are also left out of the denominator,
    since they carry no content."""
    visible = [ch for ch in (text or "") if unicodedata.category(ch) != "Cf"]
    if not visible:
        return False
    return sum(_counts_as_foreign_script(ch) for ch in visible) / len(visible) > threshold


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
    raw_text = row.get("human_text") or ""
    # Judge the article by its body: the category tags get stripped (see
    # strip_category_tags), so a page must still clear min_words without them.
    text = strip_category_tags(raw_text)
    title = row.get("title") or ""

    checks = {
        "not_category_only": not is_category_only(raw_text),
        "not_disambiguation": not is_disambiguation(text),
        "sufficient_length": word_count(text) >= min_words,
        "no_residual_markup": not has_residual_markup(text),
        "no_encoding_corruption": not (has_encoding_corruption(text) or has_encoding_corruption(title)),
        "sufficient_sinhala_content": not has_insufficient_sinhala(text),
    }
    checks["passed"] = all(checks.values())
    return checks


def filter_by_date(df: pd.DataFrame, cutoff: pd.Timestamp = GPT_ERA_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only rows both created AND last-edited before `cutoff` (see GPT_ERA_CUTOFF).

    Requires created_date/last_modified_date columns — run this on
    wikipedia_dated.parquet (build_wikipedia_dated_dataset.py), not the plain
    wikipedia.parquet, which has neither column. A row with either date missing (17 of
    10,445 rows: pages deleted/merged upstream since the dump, so the MediaWiki API had
    nothing to return) is rejected too — its authorship epoch can't be verified, so it's
    treated the same as content we can't trust rather than silently kept.
    """
    passed = (
        df["created_date"].notna()
        & df["last_modified_date"].notna()
        & (df["created_date"] < cutoff)
        & (df["last_modified_date"] < cutoff)
    )
    clean = df[passed].reset_index(drop=True)
    rejected = df[~passed].reset_index(drop=True)
    return clean, rejected


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
    clean["human_text"] = clean["human_text"].map(strip_category_tags)
    rejected = combined[~passed_mask].reset_index(drop=True)
    return clean, rejected
