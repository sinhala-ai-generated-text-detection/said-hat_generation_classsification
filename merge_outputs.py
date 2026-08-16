#!/usr/bin/env python3
"""Merge all per-domain/model JSONL outputs into a single validated dataset.

Run once, after every team member's claimed domain x model combination is complete:

    python merge_outputs.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MERGED_PATH = OUTPUTS_DIR / "merged_dataset.jsonl"

REQUIRED_FIELDS = [
    "text_id", "text", "label", "generator", "domain",
    "source_id", "prompt_id", "generation_mode",
    "input_tokens", "output_tokens",
]


def find_jsonl_files():
    return sorted(p for p in OUTPUTS_DIR.glob("*.jsonl") if p.name != MERGED_PATH.name)


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
    counts = Counter((r["domain"], r["generator"], r["generation_mode"]) for r in records)
    rows = [{"domain": d, "model": m, "mode": mo, "count": c} for (d, m, mo), c in counts.items()]
    if not rows:
        print("No records to summarize.")
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["domain", "mode"], columns="model", values="count", fill_value=0)
    print("\n=== domain x model x mode counts (gaps show up as 0 or missing columns) ===")
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

    all_valid = []
    total_invalid = 0
    for path in files:
        valid, invalid = load_and_validate(path)
        print(f"{path.name}: {len(valid)} valid, {invalid} invalid")
        all_valid.extend(valid)
        total_invalid += invalid

    with open(MERGED_PATH, "w", encoding="utf-8") as f:
        for rec in all_valid:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nMerged {len(all_valid)} records ({total_invalid} invalid skipped) -> {MERGED_PATH}")
    summarize(all_valid)
    flag_duplicates(all_valid)


if __name__ == "__main__":
    main()
