#!/usr/bin/env python3
"""Source-item content filtering, applied before allocation.

Runs on the raw records from load_source_data(), before allocate_source_items() in
allocation.py shuffles and slices them into (mode, variant) groups. This has to happen
upstream of the shuffle rather than after: the seeded shuffle/slice in
allocate_source_items depends on the exact list passed in, so filtering afterward would
reshuffle the whole item -> (mode, variant) mapping instead of just dropping bad rows.
"""


def filter_source_items(source_items: list, domain: str) -> list:
    """Drop source rows with missing/empty content for `domain` before they reach
    allocation (e.g. Wikipedia articles with no human_text).

    TODO: not implemented yet — currently a no-op. Once implemented, delete and
    regenerate config/mode_allocations/{domain}.json for every affected domain (see
    the TODO in allocation.py's get_or_create_allocation) so the cached allocation
    reflects the filtered item list rather than the original one.
    """
    return source_items
