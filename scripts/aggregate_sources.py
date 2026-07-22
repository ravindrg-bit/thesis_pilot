"""
aggregate_sources.py — fold per-URL source JSONs into the silver sources parquet.

Stage 2 of 2 (pairs with scripts/build_sources.py). build_sources.py writes one
JSON per fetched URL to data/{profile}/silver/sources_bronze/<hash>.json; this
script reads them all, assembles the single 8-column sources table, dedupes on
url_canonical, and writes:

    data/{profile}/silver/sources_{profile}.parquet

Splitting assembly from fetching means a killed fetch never loses work and the
cheap parquet build can be re-run independently. No network — pure local read.

Run:
    THESIS_RUN_PROFILE=pilot   python scripts/aggregate_sources.py
    THESIS_RUN_PROFILE=scaleup python scripts/aggregate_sources.py
"""

import json
import os
import sys
from pathlib import Path

# --- project root on sys.path (house pattern) -------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- require an EXPLICIT profile (same gate as build_silver.py) -------------
# thesis_config silently defaults THESIS_RUN_PROFILE to "pilot"; we refuse that
# implicit default here so the target data tree is always a conscious choice.
RUN_PROFILE = os.environ.get("THESIS_RUN_PROFILE")
if not RUN_PROFILE:
    sys.stderr.write(
        "[STOP] THESIS_RUN_PROFILE is not set — refusing to aggregate under an\n"
        "       implicit profile (would silently target the pilot tree).\n"
        "       Set it explicitly, e.g.:\n"
        "         THESIS_RUN_PROFILE=pilot   python scripts/aggregate_sources.py\n"
        "         THESIS_RUN_PROFILE=scaleup python scripts/aggregate_sources.py\n"
    )
    sys.exit(1)

from src.env import load_env
load_env()

import pandas as pd

import thesis_config as cfg

# 8-column schema, byte-identical to the old single-pass build_sources.py parquet.
COLS = [
    "url_canonical", "domain", "fetch_status", "http_status_code",
    "content_length", "cleaned_text", "fetch_timestamp_utc", "title_from_html",
]


def main() -> None:
    bronze_dir = cfg.SILVER / "sources_bronze"
    out_path = cfg.SILVER / f"sources_{cfg.RUN_PROFILE}.parquet"

    print("=" * 72)
    print(f"AGGREGATE SOURCES  profile={cfg.RUN_PROFILE}")
    print(f"  bronze in : {bronze_dir}")
    print(f"  parquet   : {out_path}")
    print("=" * 72)

    if not bronze_dir.exists():
        print(f"[FAIL] sources_bronze dir not found: {bronze_dir}")
        print("       Run scripts/build_sources.py first.")
        sys.exit(1)

    files = sorted(bronze_dir.glob("*.json"))
    if not files:
        print(f"[FAIL] no JSON files in {bronze_dir} — run scripts/build_sources.py first.")
        sys.exit(1)
    print(f"JSON files found : {len(files)}")

    # --- read every JSON, tolerating partial/corrupt files -------------------
    rows, bad = [], 0
    for f in files:
        try:
            rows.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            bad += 1
            if bad <= 10:
                print(f"  [skip] unreadable {f.name}: {type(e).__name__}: {e}")
    if bad:
        print(f"[WARN] skipped {bad} unreadable/partial JSON file(s)")
    if not rows:
        print("[FAIL] no readable JSON rows — nothing to aggregate.")
        sys.exit(1)

    # --- assemble the 8-column table -----------------------------------------
    df = pd.DataFrame(rows)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        print(f"[FAIL] JSON records are missing expected column(s): {missing}")
        sys.exit(1)
    df = df[COLS]
    df = df.astype({
        "http_status_code": "int64",
        "content_length":   "int64",
    })

    before = len(df)
    df = df.drop_duplicates(subset=["url_canonical"]).reset_index(drop=True)
    print(f"Rows assembled   : {before} -> {len(df)} after dedupe on url_canonical")

    # --- write parquet, fall back to CSV on any failure ----------------------
    cfg.SILVER.mkdir(parents=True, exist_ok=True)
    wrote = None
    try:
        df.to_parquet(out_path, index=False)
        wrote = out_path
        print(f"[OK] wrote parquet : {out_path}  ({len(df)} rows)")
    except Exception as e:
        print(f"[ERROR] to_parquet failed: {type(e).__name__}: {e}")
        csv_path = out_path.with_suffix(".csv")
        print(f"        falling back to CSV (may be large): {csv_path}")
        try:
            df.to_csv(csv_path, index=False)
            wrote = csv_path
            print(f"[OK] wrote CSV fallback : {csv_path}  ({len(df)} rows)")
        except Exception as e2:
            print(f"[FATAL] CSV fallback also failed: {type(e2).__name__}: {e2}")
            sys.exit(1)

    # --- brief summary -------------------------------------------------------
    ok = int((df["fetch_status"] == "ok").sum())
    print("-" * 72)
    print(f"ok rows   : {ok} ({100 * ok / len(df):.1f}%)")
    print(f"failed    : {len(df) - ok} ({100 * (len(df) - ok) / len(df):.1f}%)")
    print("status breakdown:")
    for status, count in df["fetch_status"].value_counts().items():
        print(f"  {status:<25}: {count}")
    print(f"\nWritten: {wrote}")
    print("=" * 72)


if __name__ == "__main__":
    main()
