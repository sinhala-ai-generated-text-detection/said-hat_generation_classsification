#!/usr/bin/env python3
"""Merge per-domain/model JSONL outputs into one validated dataset per domain.

Run once, after every team member's claimed domain x model combination is complete:

    python merge_outputs.py

Produces outputs/merged_<domain>.jsonl for each domain present (e.g.
merged_news.jsonl, merged_social_media.jsonl) rather than one combined file, since
each domain is its own dataset.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

REQUIRED_FIELDS = [
    "text_id", "text", "label", "generator", "domain",
    "source_id", "prompt_id", "generation_mode",
    "input_tokens", "output_tokens",
]


def merged_path_for(domain):
    return OUTPUTS_DIR / f"merged_{domain}.jsonl"


def find_jsonl_files():
    # Skip our own derived outputs, not just merged_*: human_<domain>.jsonl (built by
    # build_human_datasets.py from a *previous* merged_<domain>.jsonl) is label=0 human
    # rows in the same schema as a raw {domain}__{model}.jsonl file — without this
    # exclusion it would get swept back in here and merged into merged_<domain>.jsonl
    # alongside the AI rows, contradicting this module's own docstring ("merged_<domain>
    # .jsonl only has AI text, never human text").
    return sorted(
        p for p in OUTPUTS_DIR.glob("*.jsonl")
        if not p.name.startswith("merged_") and not p.name.startswith("human_")
    )


def load_and_validate(path):
    """Return (valid_records, invalid_count) for one JSONL file."""
    valid, invalid = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [{path.name}:{line_no}] invalid JSON, skipping")
                invalid += 1
                continue

            missing = [k for k in REQUIRED_FIELDS if k not in rec]
            if missing:
                print(f"  [{path.name}:{line_no}] missing fields {missing}, skipping")
                invalid += 1
                continue

            valid.append(rec)
    return valid, invalid


def summarize(records):
    counts = Counter((r["generator"], r["generation_mode"]) for r in records)
    rows = [{"model": m, "mode": mo, "count": c} for (m, mo), c in counts.items()]
    if not rows:
        print("No records to summarize.")
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="mode", columns="model", values="count", fill_value=0)
    print("\n=== model x mode counts (gaps show up as 0 or missing columns) ===")
    print(pivot.to_string())


def flag_duplicates(records):
    seen = defaultdict(list)
    for r in records:
        seen[r["text"]].append(r["text_id"])
    dupes = {text: ids for text, ids in seen.items() if len(ids) > 1}

    if not dupes:
        print("\nNo exact-duplicate text values found.")
        return

    print(f"\n=== {len(dupes)} exact-duplicate text values found (NOT removed — review manually) ===")
    for text, ids in list(dupes.items())[:20]:
        preview = text[:60].replace("\n", " ")
        print(f"  {ids} share identical text: {preview}...")
    if len(dupes) > 20:
        print(f"  ...and {len(dupes) - 20} more")


def main():
    files = find_jsonl_files()
    if not files:
        print(f"No .jsonl files found in {OUTPUTS_DIR}")
        return

    by_domain = defaultdict(list)
    total_invalid = 0
    for path in files:
        valid, invalid = load_and_validate(path)
        print(f"{path.name}: {len(valid)} valid, {invalid} invalid")
        for rec in valid:
            by_domain[rec["domain"]].append(rec)
        total_invalid += invalid

    for domain in sorted(by_domain):
        records = by_domain[domain]
        merged_path = merged_path_for(domain)
        with open(merged_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\n=== domain={domain}: merged {len(records)} records -> {merged_path} ===")
        summarize(records)
        flag_duplicates(records)

    print(f"\n{total_invalid} invalid record(s) skipped across all files.")


if __name__ == "__main__":
    main()
