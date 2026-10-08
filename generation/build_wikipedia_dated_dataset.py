#!/usr/bin/env python3
"""Build a quality-filtered Wikipedia source dataset with per-article revision dates.

source_data/wikipedia-monthly/wikipedia.parquet carries no timestamps at all (columns:
id, url, title, raw_mediawiki, text), so there is no way to tell a 2006 human-written
article from a recently bulk-imported/machine-translated one. Its `id` column is the real
si.wikipedia.org page id, though, so the dates can be fetched from the MediaWiki API.

This script:
  1. loads the parquet exactly the way allocation.py does (load_source_data, which renames
     text -> human_text and assigns source_id from the row position in the FULL file — so
     source_id stays consistent with config/mode_allocations/*.json),
  2. runs wikipedia_quality_filter.filter_source_data() over it (10,445 of 26,470 rows pass
     at min_words=100),
  3. adds `created_date` (timestamp of the article's FIRST revision) and
     `last_modified_date` (timestamp of its latest revision) columns,
  4. fills both from https://si.wikipedia.org/w/api.php and writes the result to
     source_data/wikipedia-monthly/wikipedia_dated.parquet.

Two passes are needed because the API only accepts `rvlimit` for a single page: latest
revisions come back 50 pages per request, but first revisions need one request per page
(hence --workers). Every fetched page is appended to a JSONL cache as it arrives, so an
interrupted run resumes instead of re-fetching.

    python build_wikipedia_dated_dataset.py            # full run (~10.4k pages)
    python build_wikipedia_dated_dataset.py --limit 50 # smoke-test on 50 pages first

Caveat worth remembering when filtering on these dates: `last_modified_date` is the live
latest revision, which is newer than the dump this parquet was built from, and
`created_date` only says when the page was created — a page created in 2008 can still have
been rewritten (or machine-translated) in 2021. Use created_date as the primary "is this
old human prose" signal and last_modified_date to spot articles that were heavily reworked
after the fact.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from generate import load_source_data
from wikipedia_quality_filter import filter_source_data

BASE_DIR = Path(__file__).resolve().parent
WIKI_DIR = BASE_DIR / "source_data" / "wikipedia-monthly"
SOURCE_PATH = WIKI_DIR / "wikipedia.parquet"
OUTPUT_PATH = WIKI_DIR / "wikipedia_dated.parquet"
CACHE_PATH = WIKI_DIR / "revision_dates_cache.jsonl"

API_URL = "https://si.wikipedia.org/w/api.php"
# Wikimedia's UA policy asks for a descriptive agent identifying the tool and its owner.
USER_AGENT = (
    "sinhala-ai-text-generation/1.0 (academic research dataset build; "
    "https://github.com/sinhala-ai-generated-text-detection/sinhala-ai-text-generation)"
)
BATCH_SIZE = 50    # API caps `pageids` at 50 per request for anonymous clients
MAXLAG = 5         # be a polite client: back off when the replicas are lagging
MAX_RETRIES = 5

CREATED_COL = "created_date"
MODIFIED_COL = "last_modified_date"

_thread_local = threading.local()
_cache_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# MediaWiki API
# --------------------------------------------------------------------------- #

def _session() -> requests.Session:
    """One requests.Session per worker thread (Session isn't thread-safe)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = session
    return session


def api_get(params: dict) -> dict:
    """GET the API with retries on transient failures (network errors, 429/503, maxlag)."""
    params = {**params, "format": "json", "formatversion": "2", "maxlag": MAXLAG}
    delay = 1.0
    last_error = None

    for _ in range(MAX_RETRIES):
        retry_after = delay
        try:
            resp = _session().get(API_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if resp.status_code in (429, 503):
                last_error = f"HTTP {resp.status_code}"
                retry_after = float(resp.headers.get("Retry-After", delay))
            elif not resp.ok:
                resp.raise_for_status()
            else:
                data = resp.json()
                error = data.get("error")
                if error is None:
                    return data
                if error.get("code") != "maxlag":
                    raise RuntimeError(f"MediaWiki API error: {error}")
                last_error = error
                retry_after = float(resp.headers.get("Retry-After", delay))

        time.sleep(retry_after)
        delay = min(delay * 2, 30.0)

    raise RuntimeError(f"MediaWiki API request failed after {MAX_RETRIES} attempts: {last_error}")


def fetch_last_modified(page_ids: list[str]) -> dict[str, str | None]:
    """Latest-revision timestamp for up to BATCH_SIZE pages in one request.

    A page that no longer exists (deleted/merged since the dump) maps to None rather than
    being left out, so it gets cached and never re-requested.
    """
    data = api_get({
        "action": "query",
        "prop": "revisions",
        "rvprop": "timestamp",
        "pageids": "|".join(page_ids),
    })
    query = data.get("query", {})

    results: dict[str, str | None] = {str(bad): None for bad in query.get("badpageids", [])}
    for page in query.get("pages", []):
        page_id = str(page.get("pageid", ""))
        if not page_id:
            continue
        revisions = page.get("revisions") or []
        results[page_id] = revisions[0]["timestamp"] if revisions else None

    # Anything the API silently dropped still needs an entry.
    for page_id in page_ids:
        results.setdefault(page_id, None)
    return results


def fetch_created(page_id: str) -> str | None:
    """Timestamp of a page's first revision. One page per request: `rvlimit` (needed to
    walk revisions oldest-first) is rejected when multiple pages are queried."""
    data = api_get({
        "action": "query",
        "prop": "revisions",
        "rvprop": "timestamp",
        "rvdir": "newer",
        "rvlimit": "1",
        "pageids": page_id,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    revisions = pages[0].get("revisions") or []
    return revisions[0]["timestamp"] if revisions else None


# --------------------------------------------------------------------------- #
# Resumable cache: {"id": <pageid>, "created_date": ..., "last_modified_date": ...}
# --------------------------------------------------------------------------- #

def load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not CACHE_PATH.exists():
        return cache
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a killed run
            cache.setdefault(str(rec["id"]), {}).update(rec)
    return cache


def append_cache(handle, page_id: str, field: str, value: str | None) -> None:
    with _cache_lock:
        handle.write(json.dumps({"id": page_id, field: value}) + "\n")
        handle.flush()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def build_filtered_frame(min_words: int, limit: int | None) -> pd.DataFrame:
    df = load_source_data(SOURCE_PATH, domain="wikipedia")
    clean_df, rejected_df = filter_source_data(df, min_words=min_words)
    print(f"{len(clean_df)} of {len(df)} rows passed the quality filter "
          f"({len(rejected_df)} rejected, min_words={min_words}).")

    clean_df = clean_df.reset_index(drop=True)
    if limit is not None:
        clean_df = clean_df.head(limit).copy()
        print(f"--limit {limit}: fetching dates for the first {len(clean_df)} rows only.")

    clean_df[CREATED_COL] = pd.NaT
    clean_df[MODIFIED_COL] = pd.NaT
    return clean_df


def fill_dates(df: pd.DataFrame, cache: dict[str, dict], workers: int) -> dict[str, dict]:
    """Fetch whatever the cache is missing, appending each result to the cache file."""
    page_ids = [str(pid) for pid in df["id"].tolist()]

    missing_modified = [pid for pid in page_ids if MODIFIED_COL not in cache.get(pid, {})]
    missing_created = [pid for pid in page_ids if CREATED_COL not in cache.get(pid, {})]
    print(f"to fetch: {len(missing_modified)} last-modified, {len(missing_created)} created "
          f"({len(page_ids) - len(missing_created)} already cached)")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with open(CACHE_PATH, "a", encoding="utf-8") as handle:
        # Pass 1 — latest revision, 50 pages per request.
        for offset in range(0, len(missing_modified), BATCH_SIZE):
            batch = missing_modified[offset:offset + BATCH_SIZE]
            for page_id, timestamp in fetch_last_modified(batch).items():
                cache.setdefault(page_id, {})[MODIFIED_COL] = timestamp
                append_cache(handle, page_id, MODIFIED_COL, timestamp)
            done = min(offset + BATCH_SIZE, len(missing_modified))
            print(f"  last-modified: {done}/{len(missing_modified)} "
                  f"({time.time() - started:.0f}s)", flush=True)

        # Pass 2 — first revision, one request per page, run concurrently.
        if missing_created:
            started = time.time()
            completed = 0

            def worker(page_id: str) -> tuple[str, str | None]:
                return page_id, fetch_created(page_id)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for page_id, timestamp in pool.map(worker, missing_created):
                    cache.setdefault(page_id, {})[CREATED_COL] = timestamp
                    append_cache(handle, page_id, CREATED_COL, timestamp)
                    completed += 1
                    if completed % 250 == 0 or completed == len(missing_created):
                        elapsed = time.time() - started
                        rate = completed / elapsed if elapsed else 0
                        remaining = (len(missing_created) - completed) / rate if rate else 0
                        print(f"  created: {completed}/{len(missing_created)} "
                              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)", flush=True)

    return cache


def apply_cache(df: pd.DataFrame, cache: dict[str, dict]) -> pd.DataFrame:
    page_ids = df["id"].astype(str)
    for column in (CREATED_COL, MODIFIED_COL):
        values = page_ids.map(lambda pid: cache.get(pid, {}).get(column))
        df[column] = pd.to_datetime(values, format="ISO8601", utc=True, errors="coerce")
    return df


def print_summary(df: pd.DataFrame) -> None:
    created = df[CREATED_COL]
    known = created.notna().sum()
    print(f"\ncreated_date resolved for {known}/{len(df)} rows "
          f"({len(df) - known} pages missing/deleted upstream).")

    if known:
        by_year = created.dt.year.value_counts().sort_index()
        print("\narticles by creation year:")
        for year, count in by_year.items():
            print(f"  {int(year)}: {count}")

        for cutoff in (2014, 2020):
            pre = (created < pd.Timestamp(f"{cutoff}-01-01", tz="UTC")).sum()
            both = ((created < pd.Timestamp(f"{cutoff}-01-01", tz="UTC"))
                    & (df[MODIFIED_COL] < pd.Timestamp(f"{cutoff}-01-01", tz="UTC"))).sum()
            print(f"\ncreated before {cutoff}: {pre} rows"
                  f"  |  created AND last edited before {cutoff}: {both} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-words", type=int, default=100,
                        help="quality-filter length floor (default: 100, matching generation_config.yaml)")
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent API requests for the per-page creation-date pass (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N filtered rows (for a smoke test)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"output parquet path (default: {OUTPUT_PATH.relative_to(BASE_DIR)})")
    args = parser.parse_args()

    if not SOURCE_PATH.exists():
        sys.exit(f"ERROR: source file not found: {SOURCE_PATH}")

    df = build_filtered_frame(args.min_words, args.limit)
    cache = fill_dates(df, load_cache(), args.workers)
    df = apply_cache(df, cache)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nwrote {len(df)} rows x {len(df.columns)} columns -> {args.output}")

    print_summary(df)


if __name__ == "__main__":
    main()
