# SinhalaAIDetection — AI text generation pipeline

Generates Sinhala AI text samples across 4 domains (`news`, `wikipedia`, `social_media`, `qa`)
using 3 models via [Amazon Bedrock](https://aws.amazon.com/bedrock/), to build a Sinhala
AI-vs-human text detection dataset. Each domain x model combination is run independently by
one team member.

## Setup

1. Install dependencies (Python 3.10+ recommended):

   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your AWS credentials:

   ```
   cp .env.example .env
   # then edit .env and set:
   #   AWS_ACCESS_KEY_ID=...
   #   AWS_SECRET_ACCESS_KEY=...
   #   AWS_REGION=us-east-1   (a region where your account has Bedrock model access)
   ```

   Already using an AWS profile or SSO? Leave the keys unset — boto3 falls back to its
   default credential chain (profile, IAM role, etc.). Either way, the account/role needs
   `bedrock:InvokeModel` permission, and each model must be individually enabled for it in
   the [Bedrock model access console](https://console.aws.amazon.com/bedrock/home#/modelaccess).

3. Add your source data files to `source_data/` — see `config/generation_config.yaml` for the
   expected filename per domain. `.csv`, `.json`, and `.parquet` are all supported.

## Before you run anything

**Claim your domain x model combination on the shared tracking sheet first:**
[TRACKING SHEET LINK — TODO: paste link here]

This avoids two people generating the same combination twice and burning API budget on
duplicate work. Mark your row as "in progress" before running, and "done" once `generate.py`
finishes.

## Running the fluency/prompt pilot

Before starting full-scale generation, run:

```
python run_pilot.py
```

For each domain, this draws a small number of random source items (`pilot.samples_per_domain`
in the config, default 2 — override with `--samples-per-domain`) straight from that domain's
source file, and runs EACH sampled item through EVERY mode x prompt-variant template for that
domain, against every model — so you're comparing phrasings on identical input rather than on
different items. Results are written to `logs/pilot_results.jsonl`. It does **not** touch the
production `outputs/` files, and re-running resamples fresh items each time (pass `--seed` for a
reproducible sample, e.g. to compare models on the exact same items).

To iterate on one domain's prompts at a time instead of testing everything:

```
python run_pilot.py --domain news
```

Review `logs/pilot_results.jsonl` manually (or knock together a quick review notebook) for each
prompt/model combination — check that the Sinhala reads fluently and that each mode is actually
doing what it's supposed to (e.g. `mode_b` continuing from `human_prefix` rather than ignoring
it). Once a variant looks right for a given mode, move its template to index 0 in
`config/generation_config.yaml`'s `prompts` section — `generate.py` always uses variant 0 for
full-scale runs. Only move on to full-scale `generate.py` runs once the pilot results look
acceptable.

## Running generation

```
python generate.py --domain news --model gemma-4-31b-it
```

- `--domain` — one of `news`, `wikipedia`, `social_media`, `qa` (from `generation_config.yaml`)
- `--model` — one of `gpt-4o`, `gemma-4-31b-it`, `claude-sonnet-5` (short names from the config)

Each model's `model_id` in `config/generation_config.yaml` must be a real Bedrock model ID
(or cross-region inference profile ID) enabled for your AWS account — `generate.py` and
`run_pilot.py` both fail fast (or skip, for the pilot) on the `TODO_BEDROCK_MODEL_ID`
placeholders until those are filled in.

Output is written incrementally to `outputs/{domain}__{model}.jsonl`, one JSON record per
line, flushed after every successful API call. Progress is also logged to
`logs/progress_tracker.csv`.

### Resuming an interrupted run

Just re-run the same command. `generate.py` reads whatever's already in the output file and
skips any (source item, mode, prompt) combination that's already been generated — no need to
track where you left off manually, and no risk of re-billing finished items.

### Cost estimates

The script prints a running estimated cost (from a hardcoded per-token pricing table in
`generate.py`) as it goes, plus a total at the end. These are approximate — check
https://aws.amazon.com/bedrock/pricing/ for current rates if you need exact figures.

## After everyone's done

Once every domain x model combination on the tracking sheet is marked complete, run:

```
python merge_outputs.py
```

This validates every record in `outputs/*.jsonl`, concatenates them into
`outputs/merged_dataset.jsonl`, prints a domain x model x mode count table so gaps are easy to
spot, and flags (without removing) any exact-duplicate `text` values for manual review.

## Filling in prompts

Each mode under `prompts:` in `config/generation_config.yaml` holds a **list** of prompt
variants (index 0, 1, 2, ...), so `run_pilot.py` can compare alternate phrasings side by side.
Replace each `TODO_FILL_IN_PROMPT` string with a finalized Sinhala prompt template. `generate.py`
only requires variant 0 (the production variant) to be filled in per mode — `run_pilot.py`
simply skips any variant still marked TODO. Available placeholders: `{title}`, `{human_text}`,
`{caption}`, `{question}`, `{human_answer}`, `{human_prefix}`.
