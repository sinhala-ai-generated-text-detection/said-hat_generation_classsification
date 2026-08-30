#!/usr/bin/env python3
"""Generate Sinhala AI text samples for one domain x model combination via OpenRouter.

Usage:
    python generate.py --domain news --model gemini-2.5-pro

Safe to interrupt and re-run: already-generated (source_id, mode, prompt_id) combinations
in the output JSONL are skipped, so a killed run just picks up where it left off.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

if sys.platform == "win32":
    import msvcrt

    def _lock_progress_file(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_progress_file(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock_progress_file(f):
        fcntl.flock(f, fcntl.LOCK_EX)

    def _unlock_progress_file(f):
        fcntl.flock(f, fcntl.LOCK_UN)

from allocation import get_or_create_allocation
from wikipedia_quality_filter import word_count

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "generation_config.yaml"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
PROGRESS_CSV = LOGS_DIR / "progress_tracker.csv"
PROGRESS_LOCK = LOGS_DIR / "progress_tracker.lock"
MAX_OUTPUT_TOKENS = 8192  # generous cap for ~250-500 word articles
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

PLACEHOLDER_KEYS = ["title", "human_text", "caption", "question", "human_answer", "human_prefix"]

# Fallback target word count for the rare mode_a item whose paired human reference field
# is empty/missing — see compute_target_word_range().
DEFAULT_TARGET_WORD_COUNT = 200
TARGET_WORD_COUNT_MARGIN = 0.2  # +/-20% band around the human reference's word count

# Some source files use column names that don't match PLACEHOLDER_KEYS (e.g. the q&a
# xlsx uses "Questions"/"Human Answer"). Map raw column -> placeholder key per domain
# so build_prompt's item.get(k, "") lookups don't silently come back empty.
COLUMN_RENAMES = {
    "qa": {"Questions": "question", "Human Answer": "human_answer"},
    "news": {"Headline": "title", "News Content": "human_text"},
    "wikipedia": {"text": "human_text"},
    "social_media": {"Message": "caption"},
}
PROGRESS_FIELDS = ["domain", "model", "completed_count", "target_count", "status", "last_updated"]

# Approximate USD-per-token pricing (input, output), per https://openrouter.ai/models as of
# 2026-08-28 — check there for current rates before trusting these for real billing.
PRICING = {
    "gemini-2.5-pro": {"input": 0.00000125, "output": 0.00001},
    "deepseek-v3": {"input": 0.0000002574, "output": 0.000001029},
    "gpt-4o": {"input": 0.0000025, "output": 0.00001},
}


# --------------------------------------------------------------------------- #
# Config / setup
# --------------------------------------------------------------------------- #

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_openrouter_session():
    """A requests.Session pre-loaded with the auth header, reused across calls."""
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: OPENROUTER_API_KEY not set.\n"
            "Add it to .env in the project root before running."
        )
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


def assert_prompts_filled(config, domain):
    """Fail fast rather than burning API calls on a TODO placeholder prompt.

    Production runs now draw items across every variant of a mode (per the mode's
    allocation, see allocation.py), so every variant needs to be filled in — not just
    index 0.
    """
    for mode, variants in config["prompts"][domain].items():
        for variant_index, template in enumerate(variants):
            if "TODO_FILL_IN_PROMPT" in template:
                sys.exit(
                    f"ERROR: prompts.{domain}.{mode}[{variant_index}] in "
                    "generation_config.yaml is still a TODO.\n"
                    "Fill in the finalized Sinhala prompt template before running."
                )


def assert_model_id_filled(model_short, model_id):
    """Fail fast rather than burning an API call on a TODO placeholder model_id."""
    if "TODO" in model_id:
        sys.exit(
            f"ERROR: models.{model_short} in generation_config.yaml is still a TODO.\n"
            "Fill in a real OpenRouter model slug before running."
        )


# --------------------------------------------------------------------------- #
# Source data
# --------------------------------------------------------------------------- #

def load_source_data(path, domain=None):
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: source data file not found: {path}")

    if path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        sys.exit(f"ERROR: unsupported source data extension '{path.suffix}' for {path}")

    if domain in COLUMN_RENAMES:
        df = df.rename(columns=COLUMN_RENAMES[domain])

    if "source_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["source_id"] = df.index.astype(str)
    return df


# --------------------------------------------------------------------------- #
# Resumability
# --------------------------------------------------------------------------- #

def output_path_for(domain, model):
    return OUTPUTS_DIR / f"{domain}__{model}.jsonl"


def load_completed_keys(output_path):
    """(source_id, mode, prompt_id) triples already present in the output file."""
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a killed run
            completed.add((rec["source_id"], rec["generation_mode"], rec["prompt_id"]))
    return completed


def existing_cost(output_path, model_short):
    """Cost already spent in prior runs, so the running total stays accurate on resume."""
    if not output_path.exists():
        return 0.0
    total = 0.0
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += estimate_cost(model_short, rec.get("input_tokens", 0), rec.get("output_tokens", 0))
    return total


def append_result(output_path, record):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())  # survive a hard kill, not just a normal crash


# --------------------------------------------------------------------------- #
# Prompting / API
# --------------------------------------------------------------------------- #

def compute_target_word_range(item, margin=TARGET_WORD_COUNT_MARGIN):
    """A +/-`margin` word-count band around the item's paired human reference
    (human_text/human_answer/caption — whichever is present), for mode_a prompts.

    mode_a items don't otherwise see the reference content (that's the point — it's
    open-ended generation from just a title/question), but the source row's other
    column is still carried on `item` alongside title/question. Using its length to
    set a per-item target — instead of one fixed word range for every item — keeps
    AI output length distributed like human text instead of systematically longer,
    which would otherwise hand a classification model trained on this data a trivial
    length shortcut instead of a real stylistic one.
    """
    reference = item.get("human_text") or item.get("human_answer") or item.get("caption") or ""
    target = word_count(reference) or DEFAULT_TARGET_WORD_COUNT
    low = max(20, round(target * (1 - margin)))
    high = round(target * (1 + margin))
    return low, high


def build_prompt(template, item):
    values = {k: str(item.get(k, "")) for k in PLACEHOLDER_KEYS}
    low, high = compute_target_word_range(item)
    values["min_word_count"] = str(low)
    values["max_word_count"] = str(high)
    return template.format(**values)


def call_model(session, model_id, prompt, max_retries=3):
    """OpenRouter's chat/completions API is OpenAI-compatible; `model_id` is an
    OpenRouter model slug (e.g. "google/gemini-2.5-pro")."""
    for attempt in range(1, max_retries + 1):
        resp = session.post(
            OPENROUTER_API_URL,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_OUTPUT_TOKENS,
            },
            timeout=90,
        )
        if resp.status_code in OPENROUTER_RETRYABLE_STATUS_CODES and attempt < max_retries:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenRouter error ({resp.status_code}) after {attempt} attempt(s): {resp.text}"
            )

        data = resp.json()
        # OpenRouter can return HTTP 200 with an embedded error object instead of a real
        # completion (e.g. the upstream provider itself timed out) — treat that the same
        # as an HTTP-level failure rather than letting the KeyError below mask it.
        if "error" in data:
            error_code = data["error"].get("code")
            if error_code in OPENROUTER_RETRYABLE_STATUS_CODES and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"OpenRouter error ({error_code}) after {attempt} attempt(s): {data['error'].get('message')}"
            )

        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    raise RuntimeError("unreachable")  # pragma: no cover


def estimate_cost(model_short, input_tokens, output_tokens):
    rates = PRICING.get(model_short)
    if not rates:
        return 0.0
    return input_tokens * rates["input"] + output_tokens * rates["output"]


# --------------------------------------------------------------------------- #
# Progress tracker (safe for concurrent domain/model runs on a shared filesystem)
# --------------------------------------------------------------------------- #

def _read_progress_rows():
    if not PROGRESS_CSV.exists():
        return []
    with open(PROGRESS_CSV, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def update_progress(domain, model, completed_count, target_count, status):
    """Read-modify-write the shared progress CSV under an flock so two team members
    running different domain/model pairs at the same time can't interleave writes
    and corrupt each other's rows."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOCK.touch(exist_ok=True)

    with open(PROGRESS_LOCK, "w") as lock_f:
        _lock_progress_file(lock_f)
        try:
            rows = _read_progress_rows()
            match = next((r for r in rows if r["domain"] == domain and r["model"] == model), None)
            new_row = {
                "domain": domain,
                "model": model,
                "completed_count": str(completed_count),
                "target_count": str(target_count),
                "status": status,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if match:
                rows[rows.index(match)] = new_row
            else:
                rows.append(new_row)

            tmp_path = PROGRESS_CSV.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            tmp_path.replace(PROGRESS_CSV)  # atomic rename on the same filesystem
        finally:
            _unlock_progress_file(lock_f)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run(domain, model_short):
    config = load_config()

    if domain not in config["domains"]:
        sys.exit(f"ERROR: unknown domain '{domain}'. Choices: {list(config['domains'])}")
    if model_short not in config["models"]:
        sys.exit(f"ERROR: unknown model '{model_short}'. Choices: {list(config['models'])}")

    # Shared across every domain x model run (and cached to disk on first use), so the
    # same source item always lands in the same (mode, variant) group no matter who
    # runs generate.py or which model they're running — this is what makes the filled
    # prompt for a given source item identical across all 3 models' runs.
    allocation = get_or_create_allocation(domain, config)

    assert_prompts_filled(config, domain)

    model_id = config["models"][model_short]
    assert_model_id_filled(model_short, model_id)
    session = get_openrouter_session()
    domain_cfg = config["domains"][domain]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(domain, model_short)
    completed = load_completed_keys(output_path)
    total_cost = existing_cost(output_path, model_short)
    total_done = len(completed)
    grand_total = sum(len(items) for variants in allocation.values() for items in variants.values())

    print(f"Domain={domain}  model={model_short} ({model_id})")
    print(f"Target per domain: {domain_cfg['target_count']}  |  modes: {list(domain_cfg['modes'])}")
    print(f"Already completed: {total_done}/{grand_total}  (resuming, ${total_cost:.4f} spent so far)")

    for mode, variants in allocation.items():
        for variant_index, items in variants.items():
            template = config["prompts"][domain][mode][int(variant_index)]
            prompt_id = f"{domain}_{mode}_v{variant_index}"

            pbar = tqdm(items, desc=f"{domain}/{mode}/v{variant_index}")
            for item in pbar:
                source_id = str(item["source_id"])
                key = (source_id, mode, prompt_id)
                if key in completed:
                    continue

                prompt = build_prompt(template, item)
                try:
                    text, in_tok, out_tok = call_model(session, model_id, prompt)
                except Exception as e:
                    pbar.write(f"  FAILED source_id={source_id} mode={mode}: {e}")
                    continue  # not marked complete — will be retried on the next run

                record = {
                    "text_id": f"{domain}_{model_short}_{mode}_{source_id}",
                    "text": text,
                    "label": 1,
                    "generator": model_short,
                    "domain": domain,
                    "source_id": source_id,
                    "prompt_id": prompt_id,
                    "generation_mode": mode,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                }
                append_result(output_path, record)
                completed.add(key)
                total_done += 1
                total_cost += estimate_cost(model_short, in_tok, out_tok)

                pbar.set_postfix(done=f"{total_done}/{grand_total}", cost=f"${total_cost:.4f}")
                update_progress(domain, model_short, total_done, grand_total, "in_progress")

    update_progress(domain, model_short, total_done, grand_total, "complete")
    print(f"\nDone. {total_done}/{grand_total} records in {output_path}")
    print(f"Estimated total cost so far for this domain/model: ${total_cost:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate Sinhala AI text via OpenRouter.")
    parser.add_argument("--domain", required=True, help="Domain key from generation_config.yaml")
    parser.add_argument("--model", required=True, help="Model short name from generation_config.yaml")
    args = parser.parse_args()
    run(args.domain, args.model)


if __name__ == "__main__":
    main()
