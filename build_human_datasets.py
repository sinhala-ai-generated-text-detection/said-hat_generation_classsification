#!/usr/bin/env python3
"""Build human-content (label=0) datasets to pair with the AI outputs already merged
by merge_outputs.py.

merge_outputs.py concatenates outputs/{domain}__{model}.jsonl, but generate.py only ever
writes AI-generated records (label=1) for the human source item it was prompted with — it
never writes the human item itself. So merged_<domain>.jsonl only has AI text, never the
human text those items were generated from.

This script recovers the human side: for each source_id referenced in
outputs/merged_<domain>.jsonl, it looks up that row in the domain's source file (the same
file/columns generate.py reads via load_source_data) and writes one label=0 record per
source_id to outputs/human_<domain>.jsonl.

Run once, after merge_outputs.py:

    python build_human_datasets.py
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from generate import COLUMN_RENAMES, load_source_data
from wikipedia_quality_filter import strip_category_tags

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIG_PATH = BASE_DIR / "config" / "generation_config.yaml"

# Which column (after COLUMN_RENAMES) holds the human-authored text to emit, per domain.
TEXT_FIELD = {
    "news": "human_text",
    "wikipedia": "human_text",
    "social_media": "caption",
}


def merged_path_for(domain):
    return OUTPUTS_DIR / f"merged_{domain}.jsonl"


def human_path_for(domain):
    return OUTPUTS_DIR / f"human_{domain}.jsonl"


def load_source_ids(merged_path):
    ids = set()
    with open(merged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.add(str(rec["source_id"]))
    return ids


def stream_json_array(path, chunk=8 << 20):
    """Yield each element of a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(chunk)
        pos = buf.index("[") + 1
        while True:
            while True:  # skip whitespace/commas, refilling as needed
                while pos < len(buf) and buf[pos] in " \n\r\t,":
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(chunk)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    more = f.read(chunk)
                    if not more:
                        raise
                    buf, pos = buf[pos:] + more, 0
            pos = end
            yield obj


def load_needed_rows(source_file, domain, source_ids):
    """Same rows/columns load_source_data() would give for `source_ids`, but for .json
    sources streams the file instead of loading it. news.json is ~1.8GB, and going
    through pandas' read_json OOM-kills this script on a 16GB machine."""
    if source_file.suffix != ".json":
        return load_source_data(source_file, domain=domain)

    renames = COLUMN_RENAMES.get(domain, {})
    rows = []
    for idx, obj in enumerate(stream_json_array(source_file)):
        # source_id is the row's position in the file, exactly as load_source_data assigns it.
        if str(idx) in source_ids:
            row = {renames.get(k, k): v for k, v in obj.items()}
            row["source_id"] = str(idx)
            rows.append(row)
    return pd.DataFrame(rows)


def build_human_dataset(domain, config):
    merged_path = merged_path_for(domain)
    if not merged_path.exists():
        print(f"skip {domain}: {merged_path} not found (run merge_outputs.py first)")
        return

    source_ids = load_source_ids(merged_path)
    print(f"\n=== domain={domain}: {len(source_ids)} unique source_id(s) in {merged_path.name} ===")

    source_file = BASE_DIR / config["domains"][domain]["source_file"]
    df = load_needed_rows(source_file, domain, source_ids)
    df = df.set_index(df["source_id"].astype(str), drop=False)

    text_field = TEXT_FIELD[domain]
    records = []
    missing, empty = [], []
    for source_id in sorted(source_ids, key=lambda x: (len(x), x)):
        if source_id not in df.index:
            missing.append(source_id)
            continue
        row = df.loc[source_id]
        text = row.get(text_field)
        if domain == "wikipedia" and isinstance(text, str):
            # Category-tag lines are only ever in the human side (AI paraphrases drop them),
            # so leaving them in would hand a classifier a free "human" signal.
            text = strip_category_tags(text)
        if text is None or (isinstance(text, float) and text != text) or not str(text).strip():
            empty.append(source_id)
            continue

        records.append({
            "text_id": f"{domain}_human_{source_id}",
            "text": str(text),
            "label": 0,
            "generator": "human",
            "domain": domain,
            "source_id": source_id,
            "prompt_id": "human",
            "generation_mode": "human",
            "input_tokens": None,
            "output_tokens": None,
        })

    out_path = human_path_for(domain)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} human record(s) -> {out_path}")
    if missing:
        print(f"  {len(missing)} source_id(s) not found in {source_file.name}: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if empty:
        print(f"  {len(empty)} source_id(s) had empty/missing '{text_field}': {empty[:10]}{'...' if len(empty) > 10 else ''}")


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for domain in ["news", "social_media", "wikipedia"]:
        build_human_dataset(domain, config)


if __name__ == "__main__":
    main()
