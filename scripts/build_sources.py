"""
build_sources.py — source-content fetch layer (medallion: silver, stage 1 of 2).

Reads the citations parquet for the active RUN_PROFILE, fetches each unique
url_canonical with BeautifulSoup (same cleaning rules as src.attribution), and
writes ONE JSON FILE PER URL to:

    data/{profile}/silver/sources_bronze/<sha256(url_canonical)[:16]>.json

Each fetch is flushed to its own file the moment it completes, so a process that
is killed mid-run never loses already-fetched pages — a resume simply skips every
URL whose JSON already exists on disk. Folding these per-URL JSONs into the single

    data/{profile}/silver/sources_{profile}.parquet

is a separate, cheap, re-runnable step:  scripts/aggregate_sources.py

Run (pilot validation):
    THESIS_RUN_PROFILE=pilot   python scripts/build_sources.py
Run (scale-up):
    THESIS_RUN_PROFILE=scaleup python scripts/build_sources.py
"""

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- require an EXPLICIT profile BEFORE anything reads the config ------------
# thesis_config silently defaults THESIS_RUN_PROFILE to "pilot"; we refuse that
# implicit default here so the target data tree is always a conscious choice.
RUN_PROFILE = os.environ.get("THESIS_RUN_PROFILE")
if not RUN_PROFILE:
    sys.stderr.write(
        "[STOP] THESIS_RUN_PROFILE is not set — refusing to build sources under an\n"
        "       implicit profile (would silently target the pilot tree).\n"
        "       Set it explicitly, e.g.:\n"
        "         THESIS_RUN_PROFILE=pilot   python scripts/build_sources.py\n"
        "         THESIS_RUN_PROFILE=scaleup python scripts/build_sources.py\n"
    )
    sys.exit(1)

from src.env import load_env
load_env()

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import thesis_config as cfg
from src.attribution import html_to_text
from src.silver import canonicalise_url

# --- Constants (kept in sync with src/attribution.py via shared html_to_text) ---
_UA = {"User-Agent": "Mozilla/5.0 (thesis-research; PAWC/AIS attribution)"}
_TIMEOUT = 20
_CHAR_LIMIT = 10_000_000   # effectively unlimited — NLI step applies its own window
_MAX_WORKERS = 10
_DOMAIN_RATE_S = 0.1       # minimum seconds between requests to the same domain

# Per-URL JSON sink (medallion "sources bronze"): one <hash>.json per fetched URL.
_BRONZE_SUBDIR = "sources_bronze"

# --- Per-domain rate-limiter (threading-safe) ---------------------------------
_domain_lock = threading.Lock()
_domain_last: dict = {}     # domain -> last request timestamp


def _domain_throttle(domain: str) -> None:
    """Sleep until at least _DOMAIN_RATE_S has elapsed since the last request to domain."""
    with _domain_lock:
        last = _domain_last.get(domain, 0.0)
        wait = _DOMAIN_RATE_S - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        _domain_last[domain] = time.monotonic()


def _url_hash(url: str) -> str:
    """Deterministic, stable filename stem for a URL: first 16 hex chars of sha256."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


# --- Core fetch (mirrors attribution.py logic, extended for persistence) ------
def _fetch_one(url_canonical: str, bronze_dir: Path) -> dict:
    """Fetch + clean one URL, flush the FULL 8-field record to its own JSON file
    (sources_bronze/<hash>.json), and return only the light fields the end-of-run
    summary needs. The heavy cleaned_text is written to disk and never accumulated
    in RAM — that, plus the per-URL flush, is the whole point of this refactor."""
    domain = urlparse(url_canonical).netloc or ""
    _domain_throttle(domain)
    ts = datetime.now(timezone.utc).isoformat()

    http_status_code = 0
    fetch_status = "ok"
    cleaned_text = ""
    title_from_html = ""

    try:
        r = requests.get(url_canonical, headers=_UA, timeout=_TIMEOUT)
        http_status_code = r.status_code
        if r.status_code != 200:
            fetch_status = f"http_{r.status_code}"
        else:
            try:
                title_tag = BeautifulSoup(r.text, "html.parser").find("title")
                title_from_html = title_tag.get_text(strip=True) if title_tag else ""
                txt = html_to_text(r.text)[:_CHAR_LIMIT]
                if txt:
                    cleaned_text = txt
                    fetch_status = "ok"
                else:
                    fetch_status = "empty_response"
            except Exception:
                fetch_status = "parse_error"
    except requests.Timeout:
        fetch_status = "timeout"
    except requests.ConnectionError:
        fetch_status = "connection_error"
    except Exception as e:
        fetch_status = f"{type(e).__name__}"

    result = {
        "url_canonical":       url_canonical,
        "domain":              domain,
        "fetch_status":        fetch_status,
        "http_status_code":    http_status_code,
        "content_length":      len(cleaned_text),
        "cleaned_text":        cleaned_text,        # empty string, never null, on failure
        "fetch_timestamp_utc": ts,
        "title_from_html":     title_from_html,
    }

    # Durable + ATOMIC per-URL flush: write to <hash>.json.tmp then os.replace()
    # onto <hash>.json. A kill mid-write can therefore never leave a truncated
    # JSON that resume would skip / aggregate would choke on — the file either
    # appears complete or not at all. os.replace is atomic on the same filesystem.
    h = _url_hash(url_canonical)
    tmp_file = bronze_dir / f"{h}.json.tmp"
    out_file = bronze_dir / f"{h}.json"
    tmp_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_file, out_file)

    # Light return only — cleaned_text stays on disk, not held across the run.
    return {"fetch_status": fetch_status, "content_length": len(cleaned_text)}


def main():
    print("=" * 72)
    print(f"BUILD SOURCES  profile={cfg.RUN_PROFILE}  workers={_MAX_WORKERS}")
    print("=" * 72)

    # --- 1. Load citations parquet -------------------------------------------
    cit_path = cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet"
    if not cit_path.exists():
        print(f"[FAIL] citations parquet not found: {cit_path}")
        sys.exit(1)
    cit = pd.read_parquet(cit_path)
    all_urls = cit["url_canonical"].dropna().unique().tolist()
    print(f"Citations rows   : {len(cit)}")
    print(f"Unique urls      : {len(all_urls)}")

    # --- 2. Resume: skip URLs whose per-URL JSON already exists on disk -------
    bronze_dir = cfg.SILVER / _BRONZE_SUBDIR
    bronze_dir.mkdir(parents=True, exist_ok=True)
    existing_hashes = {p.stem for p in bronze_dir.glob("*.json")}
    todo = [u for u in all_urls if _url_hash(u) not in existing_hashes]
    print(f"Resume: {len(all_urls) - len(todo)} URLs already on disk in "
          f"{_BRONZE_SUBDIR}/ — skipping")
    print(f"To fetch         : {len(todo)}")
    print("-" * 72)

    # --- 3. Parallel fetch — each worker flushes its own JSON -----------------
    if todo:
        summary_rows: list = []
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_fetch_one, url, bronze_dir): url for url in todo}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="fetching", unit="url", dynamic_ncols=True):
                summary_rows.append(fut.result())
        _print_summary(summary_rows, all_urls)
    else:
        print("Nothing to fetch — every URL already has a JSON on disk.")

    # --- 4. Point at the data (aggregation is a separate step) ---------------
    n_files = len(list(bronze_dir.glob("*.json")))
    print(f"\nPer-URL JSON written to : {bronze_dir}  ({n_files} files on disk)")
    print(f"Next step: build the parquet with scripts/aggregate_sources.py  "
          f"(THESIS_RUN_PROFILE={cfg.RUN_PROFILE})")


def _print_summary(rows: list, all_urls: list) -> None:
    if not rows:
        print("No rows fetched this run to summarise.")
        return
    df = pd.DataFrame(rows)
    fetched = len(df)
    total_unique = len(all_urls)
    ok = int((df["fetch_status"] == "ok").sum())
    failed = fetched - ok
    pct_ok = 100 * ok / fetched if fetched else 0
    pct_fail = 100 * failed / fetched if fetched else 0

    print()
    print("=" * 72)
    print("SOURCES FETCH SUMMARY (this run)")
    print("=" * 72)
    print(f"Total unique URLs     : {total_unique}")
    print(f"Fetched this run      : {fetched}")
    print(f"  ok                  : {ok} ({pct_ok:.1f}%)")
    print(f"  failed              : {failed} ({pct_fail:.1f}%)")

    status_counts = df["fetch_status"].value_counts()
    print("\nFetch status breakdown (this run):")
    for status, count in status_counts.items():
        print(f"  {status:<25}: {count}")

    ok_rows = df[df["fetch_status"] == "ok"]
    if len(ok_rows):
        mean_len = ok_rows["content_length"].mean()
        print(f"\nMean content length   : {mean_len:,.0f} chars (ok rows only)")
    print("=" * 72)


if __name__ == "__main__":
    main()
