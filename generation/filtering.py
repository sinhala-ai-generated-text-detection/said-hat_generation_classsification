#!/usr/bin/env python3
"""Source-item content filtering, applied before allocation.

Runs on the raw records from load_source_data(), before slice_model_items()/
allocate_modes() in allocation.py shuffle and slice them into (model, mode, variant)
groups. This has to happen upstream of the shuffle rather than after: the seeded
shuffle/slice depends on the exact list passed in, so filtering afterward would
reshuffle the whole item -> (mode, variant) mapping instead of just dropping bad rows.

Motivated by contamination found in the already-generated news dataset: 28/1499 source
rows were mojibake-corrupted (double-encoded UTF-8) or genuine Tamil-language articles
that leaked into what's meant to be a Sinhala-only corpus — nothing had ever screened
news/social_media for this, since wikipedia_quality_filter.filter_source_data() (which
does check for this, via has_encoding_corruption/has_insufficient_sinhala) is only wired
into allocation.py for domain == "wikipedia". This reuses those same two checks — they
were never wikipedia-specific despite the module name — plus a per-domain minimum word
count, for every domain.
"""

from wikipedia_quality_filter import has_encoding_corruption, has_insufficient_sinhala, word_count

# Which column holds the human-authored text to check, per domain (after generate.py's
# COLUMN_RENAMES has already run — see generate.py's load_source_data).
TEXT_FIELD = {
    "news": "human_text",
    "wikipedia": "human_text",
    "social_media": "caption",
}


def filter_source_items(source_items: list, domain: str, min_words: int = 0) -> list:
    """Drop source rows with missing/empty/too-short content, encoding corruption, or
    insufficient Sinhala-script content for `domain`.

    `min_words` is the per-domain floor from generation_config.yaml (domains.<domain>
    .min_words) — 0 (no floor) if the domain doesn't set one. Wikipedia already enforces
    its own min_words via wikipedia_quality_filter.filter_source_data() upstream of this
    call, so re-checking it here is a harmless no-op for that domain, not double
    filtering against a different threshold.
    """
    field = TEXT_FIELD.get(domain)
    if not field:
        return source_items

    kept = []
    for item in source_items:
        text = item.get(field)
        if not isinstance(text, str):  # None/NaN — a row with no content at all
            continue
        if word_count(text) < min_words:
            continue
        if has_encoding_corruption(text):
            continue
        if has_insufficient_sinhala(text):
            continue
        kept.append(item)
    return kept
