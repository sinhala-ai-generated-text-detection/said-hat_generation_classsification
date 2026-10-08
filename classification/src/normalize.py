"""
Normalization pipeline applied to Sinhala-HAT before training (scripts/01_build_corpus.py).

Applies the SAME transformations to both classes (human and AI text), so that
whatever the classifier learns after normalization cannot come from a
class-specific rule baked into the cleaning code itself.

Transformations (each individually togglable, all on by default):
  1. latin_to_placeholder — every maximal run of Latin-script characters
     (a "Latin-script token") is replaced by a single <LAT> placeholder.
     Word count is preserved: one token in, one token out, never deleted.
  2. strip_emoji — removes emoji / pictographic symbols entirely (they are
     not word tokens, so removing them does not touch word count).
  3. strip_wiki_markup — removes residual MediaWiki markup tokens that can
     leak onto the HUMAN side of Wikipedia pairs (thumb, right, link=,
     image filenames, File:, Image:).
  4. normalize_punct_whitespace — collapses whitespace runs, normalizes
     quote/dash variants to a canonical form, and collapses repeated
     punctuation marks (e.g. "!!!" -> "!", "..." kept as ellipsis-safe but
     ">>>>" -> ">").

Word-count preservation is verified by `check_word_count_preserved`.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

try:
    import emoji as _emoji_lib
except ImportError:  # pragma: no cover
    _emoji_lib = None


LAT_PLACEHOLDER = "<LAT>"

# ---------------------------------------------------------------------------
# 1. Latin-script token -> placeholder
# ---------------------------------------------------------------------------
# A "Latin-script token" is a maximal run of characters drawn from the Latin
# alphabet plus the punctuation/digits that commonly glue Latin tokens
# together (dots in abbreviations/URLs, @, /, -, _, digits). We match on
# whitespace-delimited tokens: a token counts as Latin if it contains at
# least one Latin letter and NO Sinhala-block character.
_SINHALA_BLOCK = re.compile(r"[඀-෿]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_WORD_SPLIT = re.compile(r"(\s+)")  # keep whitespace so join is exact


def _is_latin_token(tok: str) -> bool:
    if not _LATIN_LETTER.search(tok):
        return False
    if _SINHALA_BLOCK.search(tok):
        return False
    return True


def latin_to_placeholder(text: str) -> str:
    """Replace every whitespace-delimited Latin-script token with <LAT>.

    Preserves word count exactly: splitting on whitespace before and after
    yields the same number of non-whitespace tokens.
    """
    parts = _WORD_SPLIT.split(text)
    out = []
    for p in parts:
        if p == "" or p.isspace():
            out.append(p)
            continue
        out.append(LAT_PLACEHOLDER if _is_latin_token(p) else p)
    return "".join(out)


def count_latin_tokens(text: str) -> int:
    return sum(1 for p in text.split() if _is_latin_token(p))


def latin_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    n_latin = sum(1 for ch in text if _LATIN_LETTER.match(ch))
    return n_latin / max(1, len(text))


# ---------------------------------------------------------------------------
# 2. Emoji stripping
# ---------------------------------------------------------------------------
if _emoji_lib is not None:
    def count_emoji(text: str) -> int:
        return _emoji_lib.emoji_count(text)

    def strip_emoji(text: str) -> str:
        return _emoji_lib.replace_emoji(text, replace="")
else:  # pragma: no cover - fallback if `emoji` package unavailable
    _EMOJI_RANGE = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002600-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "\U00002190-\U000021FF"
        "\U00002B00-\U00002BFF"
        "]+",
        flags=re.UNICODE,
    )

    def count_emoji(text: str) -> int:
        return len(_EMOJI_RANGE.findall(text))

    def strip_emoji(text: str) -> str:
        return _EMOJI_RANGE.sub("", text)


# ---------------------------------------------------------------------------
# 3. Wikipedia markup tokens
# ---------------------------------------------------------------------------
# Tokens that indicate un-rendered MediaWiki markup leaking into plain text,
# e.g. "[[File:Foo.jpg|thumb|right|Caption]]" left partially un-stripped, or
# a bare "thumb"/"right" sizing keyword, or "link=" parameters.
_WIKI_MARKUP_PATTERNS = [
    re.compile(r"\[\[[^\]]*\]\]"),                       # [[...]] wikilinks
    re.compile(r"\{\{[^}]*\}\}"),                         # {{...}} templates
    re.compile(r"\b(?:File|Image)\s*:\s*\S+", re.I),      # File:foo.jpg / Image:foo.png
    re.compile(r"\S+\.(?:jpg|jpeg|png|gif|svg|ogg|webp)\b", re.I),  # bare image filenames
    re.compile(r"\blink\s*=\s*\S*", re.I),                # link=...
    re.compile(r"\b(?:thumb|thumbnail|right|left|frameless|frame|upright)\b", re.I),
]


def has_wiki_markup(text: str) -> bool:
    return any(p.search(text) for p in _WIKI_MARKUP_PATTERNS)


def wiki_markup_token_count(text: str) -> int:
    return sum(len(p.findall(text)) for p in _WIKI_MARKUP_PATTERNS)


def strip_wiki_markup(text: str) -> str:
    out = text
    for p in _WIKI_MARKUP_PATTERNS:
        out = p.sub(" ", out)
    return out


# ---------------------------------------------------------------------------
# 4. Punctuation / whitespace / quote / dash normalization
# ---------------------------------------------------------------------------
_QUOTE_MAP = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "«": '"', "»": '"',
    "‹": "'", "›": "'",
    "`": "'",
}
_DASH_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
}
_REPEATED_PUNCT = re.compile(r"([!?.,;:\-_*#~])\1{1,}")
_WHITESPACE_RUN = re.compile(r"[ \t ​‌‍]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_punct_whitespace(text: str) -> str:
    out = unicodedata.normalize("NFC", text)
    for src, dst in _QUOTE_MAP.items():
        out = out.replace(src, dst)
    for src, dst in _DASH_MAP.items():
        out = out.replace(src, dst)
    out = _REPEATED_PUNCT.sub(r"\1", out)
    out = _WHITESPACE_RUN.sub(" ", out)
    out = _MULTI_NEWLINE.sub("\n\n", out)
    out = "\n".join(line.strip() for line in out.split("\n"))
    out = out.strip()
    return out


# ---------------------------------------------------------------------------
# Punctuation-mark breakdown
# ---------------------------------------------------------------------------
_PUNCT_MARKS = list(".,!?;:-_()[]{}\"'/\\|@#$%^&*+=<>~`")


def punct_breakdown(text: str) -> dict:
    return {m: text.count(m) for m in _PUNCT_MARKS}


def punct_density(text: str) -> float:
    n_words = max(1, len(text.split()))
    n_punct = sum(1 for ch in text if ch in _PUNCT_MARKS)
    return n_punct / n_words


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
@dataclass
class NormalizeConfig:
    latin_to_placeholder: bool = True
    strip_emoji: bool = True
    strip_wiki_markup: bool = True
    normalize_punct_whitespace: bool = True


def normalize_text(text: str, cfg: NormalizeConfig = NormalizeConfig()) -> str:
    out = text
    if cfg.strip_wiki_markup:
        out = strip_wiki_markup(out)
    if cfg.strip_emoji:
        out = strip_emoji(out)
    if cfg.latin_to_placeholder:
        out = latin_to_placeholder(out)
    if cfg.normalize_punct_whitespace:
        out = normalize_punct_whitespace(out)
    return out


def check_word_count_preserved(raw: str, transformed_latin_only: str) -> bool:
    """Sanity check for the <LAT> substitution step in isolation: word count
    must be identical between raw and latin-placeholder text (whitespace
    normalization is allowed to change counts and is checked separately)."""
    return len(raw.split()) == len(transformed_latin_only.split())


# ---------------------------------------------------------------------------
# Within-pair length truncation (applied AFTER normalization, on top of it)
# ---------------------------------------------------------------------------
def truncate_to_word_count(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[:n_words])


def length_match_pair(text_a: str, text_b: str) -> tuple[str, str]:
    """Truncate both documents in a human/AI pair to the shorter one's word
    count. Returns (new_a, new_b)."""
    n = min(len(text_a.split()), len(text_b.split()))
    return truncate_to_word_count(text_a, n), truncate_to_word_count(text_b, n)
