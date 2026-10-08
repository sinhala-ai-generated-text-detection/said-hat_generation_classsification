"""
Deterministic Sinhala sentence segmenter.

Adapted, standalone, from `boundary_detection_test_01/src/segment.py` (the
same user's sentence-boundary-detection project) — that copy is described
there as "THE segmenter" used for source prep, generation length targets,
labelling and evaluation across that pipeline. This copy drops its
config.yaml dependency (`get_segmenter`) and keeps the pure, deterministic
`SinhalaSegmenter` class and its defaults unchanged, so sentence counts here
follow the same convention.

Splitting happens on '.', '?', '!', '෴' (kunddaliya) and '…', with guards for:
  - abbreviations   (ආචාර්ය.  Dr.  ක්‍රි.ව.)
  - decimals        (3.14, 1,250.50)
  - initials        (A. B. Silva)
  - ellipsis runs   (...)
  - closing quotes/brackets that trail the terminator
"""
from __future__ import annotations

import re
from typing import Sequence

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_TRAILING = "\"'’”)]}»"
_DEFAULT_TERMINATORS = [".", "?", "!", "෴", "…"]
_DEFAULT_ABBREV: list[str] = [
    "ආචාර්ය", "මහාචාර්ය", "පූජ්‍ය", "ශ්‍රී", "මයා", "මිය", "අංක",
    "ක්‍රි.ව", "ක්‍රි.පූ", "ඉ.පෙ", "පි.ව",
    "Dr", "Mr", "Mrs", "Ms", "Prof", "St", "Jr", "Sr",
    "vs", "etc", "No", "Fig", "Vol", "pp", "ed", "eds", "cf", "al",
]


class SinhalaSegmenter:
    def __init__(
        self,
        terminators: Sequence[str] | None = None,
        abbreviations: Sequence[str] | None = None,
        min_sentence_chars: int = 15,
    ) -> None:
        self.terminators = list(terminators or _DEFAULT_TERMINATORS)
        self.abbreviations = set(abbreviations or _DEFAULT_ABBREV)
        self.min_sentence_chars = min_sentence_chars
        self._abbrev_sorted = sorted(self.abbreviations, key=len, reverse=True)
        self._term_set = set(self.terminators)

    def _is_decimal_point(self, text: str, i: int) -> bool:
        if text[i] != ".":
            return False
        prev_digit = i > 0 and text[i - 1].isdigit()
        nxt = i + 1
        next_digit = nxt < len(text) and text[nxt].isdigit()
        return prev_digit and next_digit

    def _preceding_token(self, text: str, i: int) -> str:
        j = i
        while j > 0 and not text[j - 1].isspace():
            j -= 1
        return text[j:i]

    def _is_abbreviation(self, text: str, i: int) -> bool:
        if text[i] != ".":
            return False
        tok = self._preceding_token(text, i)
        if tok in self.abbreviations:
            return True
        for ab in self._abbrev_sorted:
            if tok == ab or tok.endswith("." + ab) or tok == ab.replace(".", ""):
                return True
        if len(tok) == 1 and tok.isascii() and tok.isalpha():
            return True
        return False

    def _is_ellipsis_interior(self, text: str, i: int) -> bool:
        if text[i] != ".":
            return False
        return i + 1 < len(text) and text[i + 1] == "."

    @staticmethod
    def _quote_positions(text: str) -> set[int]:
        inside: set[int] = set()
        for quote in ('"', "“”", "‘’"):
            if len(quote) == 2:
                opens = [k for k, c in enumerate(text) if c == quote[0]]
                closes = [k for k, c in enumerate(text) if c == quote[1]]
                if len(opens) != len(closes):
                    continue
                for o, c in zip(opens, closes):
                    if o < c:
                        inside.update(range(o + 1, c))
            else:
                pos = [k for k, c in enumerate(text) if c == quote]
                if len(pos) % 2 != 0:
                    continue
                for o, c in zip(pos[0::2], pos[1::2]):
                    inside.update(range(o + 1, c))
        return inside

    def segment(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        raw_pieces: list[str] = []
        quoted = self._quote_positions(text)
        start = 0
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in self._term_set:
                if (
                    i in quoted
                    or self._is_decimal_point(text, i)
                    or self._is_abbreviation(text, i)
                    or self._is_ellipsis_interior(text, i)
                ):
                    i += 1
                    continue
                j = i + 1
                while j < n and (text[j] in self._term_set or text[j] in _TRAILING):
                    j += 1
                if j < n and not text[j].isspace():
                    i += 1
                    continue
                piece = text[start:j].strip()
                if piece:
                    raw_pieces.append(piece)
                start = j
                i = j
                continue
            if ch == "\n":
                j = i
                while j < n and text[j] in "\r\n":
                    j += 1
                piece = text[start:i].strip()
                if piece:
                    raw_pieces.append(piece)
                start = j
                i = j
                continue
            i += 1
        tail = text[start:].strip()
        if tail:
            raw_pieces.append(tail)
        return self._merge_fragments(raw_pieces)

    def _is_fragment(self, piece: str) -> bool:
        if not _LETTER_RE.search(piece):
            return True
        ends_terminated = piece.rstrip(_TRAILING).endswith(tuple(self._term_set))
        return not ends_terminated and len(piece) < self.min_sentence_chars

    def _merge_fragments(self, pieces: list[str]) -> list[str]:
        out: list[str] = []
        for p in pieces:
            if out and self._is_fragment(p):
                out[-1] = out[-1] + " " + p
            else:
                out.append(p)
        if len(out) > 1 and self._is_fragment(out[0]):
            out[1] = out[0] + " " + out[1]
            out.pop(0)
        return [re.sub(r"\s+", " ", s).strip() for s in out if s.strip()]


_SEGMENTER = SinhalaSegmenter()


def segment_sentences(text: str) -> list[str]:
    return _SEGMENTER.segment(text)


def sentence_count(text: str) -> int:
    return len(segment_sentences(text))
