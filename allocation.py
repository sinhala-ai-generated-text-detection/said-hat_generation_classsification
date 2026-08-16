#!/usr/bin/env python3
"""Deterministic, disjoint allocation of source items to (mode, variant) groups.

Each domain's source rows are shuffled once (with a fixed seed) and sliced up so that
every mode gets its configured `count` of items, and each mode's slice is further split
into `num_variants` disjoint sub-groups. No source item is ever used by more than one
(mode, variant) combination — this is what lets generate.py and run_pilot.py draw from
the same pools without double-using an item or silently overlapping across variants.

The resulting allocation is cached to config/mode_allocations/{domain}.json on first use
so every team member's run (and every model, since generate.py is invoked once per
domain x model) sees the identical item -> (mode, variant) assignment.
"""

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ALLOCATIONS_DIR = BASE_DIR / "config" / "mode_allocations"


def allocate_source_items(source_items: list, mode_config: dict, seed: int) -> dict:
    """Shuffle source_items (seeded, independent of the global `random` module) and
    slice them into disjoint per-mode, per-variant groups.

    Returns {mode_name: {variant_index: [source_items...]}}.
    """
    total_requested = sum(cfg["count"] for cfg in mode_config.values())
    if total_requested > len(source_items):
        breakdown = ", ".join(f"{mode}={cfg['count']}" for mode, cfg in mode_config.items())
        raise ValueError(
            f"Requested {total_requested} source items across modes ({breakdown}), "
            f"but only {len(source_items)} source items are available."
        )

    rng = random.Random(seed)
    shuffled = list(source_items)
    rng.shuffle(shuffled)

    allocation = {}
    cursor = 0
    for mode, cfg in mode_config.items():
        count = cfg["count"]
        num_variants = cfg["num_variants"]
        mode_items = shuffled[cursor:cursor + count]
        cursor += count
        allocation[mode] = _split_evenly(mode_items, num_variants)

    return allocation


def _split_evenly(items, num_variants):
    """Split items into num_variants disjoint sub-lists, as evenly as possible, with
    any remainder distributed to the first few variants."""
    n = len(items)
    base, remainder = divmod(n, num_variants)
    groups = {}
    start = 0
    for variant_index in range(num_variants):
        size = base + (1 if variant_index < remainder else 0)
        groups[variant_index] = items[start:start + size]
        start += size
    return groups


def verify_no_overlap(allocation: dict) -> None:
    """Assert no source_id appears in more than one (mode, variant) group."""
    seen = {}  # source_id -> [ "mode/variant_index", ... ]
    for mode, variants in allocation.items():
        for variant_index, items in variants.items():
            group_label = f"{mode}/{variant_index}"
            for item in items:
                source_id = str(item["source_id"])
                seen.setdefault(source_id, []).append(group_label)

    overlaps = {sid: groups for sid, groups in seen.items() if len(groups) > 1}
    if overlaps:
        details = "; ".join(f"{sid} appears in {groups}" for sid, groups in overlaps.items())
        raise AssertionError(f"Overlapping source_id(s) found across allocation groups: {details}")


def _is_present(value):
    if value is None:
        return False
    if isinstance(value, float) and value != value:  # NaN
        return False
    return True


def _trim_item(item, extra_fields):
    trimmed = {"source_id": str(item["source_id"])}
    for field in extra_fields:
        if field in item and _is_present(item[field]):
            trimmed[field] = item[field]
    return trimmed


def get_or_create_allocation(domain: str, config: dict) -> dict:
    """Load the cached allocation for `domain` if one exists, otherwise build it,
    verify it, save it, and return it. Re-running never re-shuffles or produces a
    different allocation once config/mode_allocations/{domain}.json exists.
    """
    path = ALLOCATIONS_DIR / f"{domain}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Deferred import: generate.py imports get_or_create_allocation at module load
    # time, so importing generate.py here eagerly would be circular. By the time this
    # function actually runs, generate.py has finished loading.
    from generate import load_source_data, PLACEHOLDER_KEYS

    domain_cfg = config["domains"][domain]
    seed = config.get("seed", 42)

    df = load_source_data(BASE_DIR / domain_cfg["source_file"], domain=domain)
    source_items = df.to_dict("records")

    allocation = allocate_source_items(source_items, domain_cfg["modes"], seed)
    verify_no_overlap(allocation)

    trimmed = {
        mode: {
            str(variant_index): [_trim_item(item, PLACEHOLDER_KEYS) for item in items]
            for variant_index, items in variants.items()
        }
        for mode, variants in allocation.items()
    }

    ALLOCATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)

    return trimmed
