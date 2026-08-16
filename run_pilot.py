#!/usr/bin/env python3
"""Small-scale pilot across every domain x mode x prompt-variant x model combination.

Run this BEFORE any full-scale generate.py run: it samples a handful of items per
combination, calls each prompt variant against each model, and writes everything to
logs/pilot_results.jsonl for manual fluency/mode-correctness review. It never touches
the production outputs/{domain}__{model}.jsonl files.

Usage:
    python run_pilot.py
    python run_pilot.py --samples-per-combination 3
"""

import argparse
import json

from tqdm import tqdm

from allocation import get_or_create_allocation
from generate import (
    BASE_DIR,
    build_prompt,
    call_model,
    estimate_cost,
    load_api_key,
    load_config,
)

PILOT_OUTPUT = BASE_DIR / "logs" / "pilot_results.jsonl"


def append_pilot_result(record):
    PILOT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PILOT_OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_pilot(config, samples_per_combination):
    api_keys = {}  # provider -> key, loaded lazily so unused providers don't need a key set

    total_cost = 0.0
    combos_tested = 0
    variants_skipped = 0

    for domain, domain_cfg in config["domains"].items():
        # Same allocation generate.py will use for full-scale runs — the pilot samples
        # from its disjoint (mode, variant_index) pools instead of sampling
        # independently, so pilot results reflect the actual items each mode/variant
        # will draw from in production.
        allocation = get_or_create_allocation(domain, config)

        for mode, variants in config["prompts"][domain].items():
            for variant_index, template in enumerate(variants):
                if "TODO_FILL_IN_PROMPT" in template:
                    print(f"SKIP {domain}/{mode}/variant{variant_index}: prompt still a TODO")
                    variants_skipped += 1
                    continue

                # Same slice of the allocation's pool reused for every model below, so
                # every model is compared against identical items for this variant.
                group_items = allocation[mode][str(variant_index)]
                items = group_items[:samples_per_combination]

                for model_short, model_cfg in config["models"].items():
                    provider = model_cfg["provider"]
                    model_id = model_cfg["model_id"]
                    if provider not in api_keys:
                        api_keys[provider] = load_api_key(provider)
                    api_key = api_keys[provider]

                    label = f"{domain}/{mode}/v{variant_index}/{model_short}"
                    pbar = tqdm(items, desc=label)
                    for item in pbar:
                        source_id = str(item["source_id"])
                        prompt = build_prompt(template, item)
                        try:
                            text, in_tok, out_tok = call_model(provider, api_key, model_id, prompt)
                        except Exception as e:
                            pbar.write(f"  FAILED source_id={source_id}: {e}")
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
                        pbar.set_postfix(cost=f"${total_cost:.4f}")

                    combos_tested += 1

    print("\nPilot complete.")
    print(f"(domain, mode, variant, model) combinations tested: {combos_tested}")
    if variants_skipped:
        print(f"Prompt variants skipped (still TODO, all models skipped for them): {variants_skipped}")
    print(f"Estimated total cost: ${total_cost:.4f}")
    print(f"Results written to: {PILOT_OUTPUT}")


def main():
    config = load_config()
    default_samples = config.get("pilot", {}).get("samples_per_combination", 2)

    parser = argparse.ArgumentParser(
        description="Run a small fluency/prompt-variant pilot via OpenRouter."
    )
    parser.add_argument(
        "--samples-per-combination",
        type=int,
        default=default_samples,
        help=(
            "Source items to sample per (domain, mode, variant, model) combination "
            f"(default: {default_samples}, from config.pilot.samples_per_combination)"
        ),
    )
    args = parser.parse_args()
    run_pilot(config, args.samples_per_combination)


if __name__ == "__main__":
    main()
