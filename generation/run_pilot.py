#!/usr/bin/env python3
"""Small-scale pilot for reviewing prompt variants, domain by domain.

Run this BEFORE any full-scale generate.py run. For each domain (or just the one named
by --domain), it draws a handful of random source items straight from that domain's
source file and runs EACH item through EVERY mode x prompt-variant template for that
domain, against every model. Testing the same item across variants (rather than a
different item per variant) is what lets you actually judge whether one phrasing reads
better than another. This is independent of generate.py's production allocation — the
pilot is for prompt review, not for reserving items for the real run — so it never
touches outputs/{domain}__{model}.jsonl and re-running can resample fresh items.

Usage:
    python run_pilot.py
    python run_pilot.py --domain news
    python run_pilot.py --domain news --samples-per-domain 3
    python run_pilot.py --domain news --seed 7   # reproducible sample for comparison runs
"""

import argparse
import json
import random

from tqdm import tqdm

from generate import (
    BASE_DIR,
    build_prompt,
    call_model,
    estimate_cost,
    get_openrouter_session,
    load_config,
    load_source_data,
)

PILOT_OUTPUT = BASE_DIR / "logs" / "pilot_results.jsonl"


def append_pilot_result(record):
    PILOT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PILOT_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def sample_domain_items(domain, domain_cfg, n, rng):
    """n random source items straight from `domain`'s source file (not the production
    allocation), so the same item can be reused across every mode/variant template."""
    df = load_source_data(BASE_DIR / domain_cfg["source_file"], domain=domain)
    items = df.to_dict("records")
    return rng.sample(items, min(n, len(items)))


def filled_variants(config, domain):
    """(mode, variant_index, template) for every non-TODO prompt variant in `domain`,
    plus a count of variants skipped for still being a TODO."""
    filled = []
    skipped = 0
    for mode, variants in config["prompts"][domain].items():
        for variant_index, template in enumerate(variants):
            if "TODO_FILL_IN_PROMPT" in template:
                print(f"SKIP {domain}/{mode}/variant{variant_index}: prompt still a TODO")
                skipped += 1
                continue
            filled.append((mode, variant_index, template))
    return filled, skipped


def filled_models(config):
    """(model_short, model_id) for every model whose model_id isn't still a TODO
    placeholder, plus a count of models skipped for that reason."""
    filled = []
    skipped = 0
    for model_short, model_id in config["models"].items():
        if "TODO" in model_id:
            print(f"SKIP model {model_short}: model_id still a TODO")
            skipped += 1
            continue
        filled.append((model_short, model_id))
    return filled, skipped


def run_pilot(config, samples_per_domain, only_domain=None, seed=None):
    if only_domain and only_domain not in config["domains"]:
        raise SystemExit(f"ERROR: unknown domain '{only_domain}'. Choices: {list(config['domains'])}")
    domains = [only_domain] if only_domain else list(config["domains"])

    rng = random.Random(seed) if seed is not None else random.Random()
    models, models_skipped = filled_models(config)
    session = get_openrouter_session()  # shared across every model, reused per call

    total_cost = 0.0
    calls_made = 0
    variants_skipped = 0

    for domain in domains:
        domain_cfg = config["domains"][domain]
        variants, skipped = filled_variants(config, domain)
        variants_skipped += skipped
        items = sample_domain_items(domain, domain_cfg, samples_per_domain, rng)

        total_calls = len(items) * len(variants) * len(models)
        pbar = tqdm(total=total_calls, desc=domain)
        for item in items:
            source_id = str(item["source_id"])

            for mode, variant_index, template in variants:
                prompt = build_prompt(template, item)

                for model_short, model_id in models:
                    try:
                        text, in_tok, out_tok = call_model(session, model_id, prompt)
                    except Exception as e:
                        pbar.write(f"  FAILED {domain}/{mode}/v{variant_index}/{model_short}/{source_id}: {e}")
                        pbar.update(1)
                        continue

                    record = {
                        "domain": domain,
                        "mode": mode,
                        "variant_index": variant_index,
                        "model": model_short,
                        "source_id": source_id,
                        "generated_text": text,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    }
                    append_pilot_result(record)

                    total_cost += estimate_cost(model_short, in_tok, out_tok)
                    calls_made += 1
                    pbar.set_postfix(cost=f"${total_cost:.4f}")
                    pbar.update(1)

    print("\nPilot complete.")
    print(f"Domains tested: {domains}")
    print(f"Calls made: {calls_made}")
    if variants_skipped:
        print(f"Prompt variants skipped (still TODO): {variants_skipped}")
    if models_skipped:
        print(f"Models skipped (model_id still TODO): {models_skipped}")
    print(f"Estimated total cost: ${total_cost:.4f}")
    print(f"Results written to: {PILOT_OUTPUT}")


def main():
    config = load_config()
    default_samples = config.get("pilot", {}).get("samples_per_domain", 2)

    parser = argparse.ArgumentParser(
        description="Run a small prompt-variant pilot, domain by domain, via each model's API."
    )
    parser.add_argument("--domain", default=None, help="Only pilot this domain (default: all domains)")
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=default_samples,
        help=(
            "Random source items sampled per domain; each one is run through every "
            f"mode x prompt-variant template for that domain (default: {default_samples}, "
            "from config.pilot.samples_per_domain)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the random sample so re-running produces the same items (default: fresh sample each run)",
    )
    args = parser.parse_args()
    run_pilot(config, args.samples_per_domain, only_domain=args.domain, seed=args.seed)


if __name__ == "__main__":
    main()
