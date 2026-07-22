"""
build_pilot_queries.py — select and FREEZE the pilot query set from GEO-Bench.

A pilot-grade slice of V3 Stage 1 (acquire/audit GEO-Bench; define the auditable
working query frame). Loads GEO-Bench, records provenance, prints a short profile
plus a couple of full records, draws cfg.N_QUERIES with a SEEDED random sample,
derives a stable query_id per row (collision-checked), and writes a FROZEN, versioned
registry (parquet) + JSON manifest + CSV preview, refusing to overwrite a frozen file.

Run from the project root:  python scripts/build_pilot_queries.py
"""

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

import thesis_config as cfg
from src.repro import set_global_seeds
from src.geobench import load_geobench, derive_query_id, profile_tags

REGISTRY_VERSION = cfg.REGISTRY_VERSION
OUT_DIR = cfg.QUERY_DIR
REGISTRY_PARQUET = cfg.QUERY_REGISTRY
REGISTRY_CSV = OUT_DIR / f"{cfg.RUN_PROFILE}_queries_{REGISTRY_VERSION}.preview.csv"
MANIFEST_JSON = OUT_DIR / f"{cfg.RUN_PROFILE}_queries_{REGISTRY_VERSION}.manifest.json"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if REGISTRY_PARQUET.exists():
        print(f"[STOP] {REGISTRY_PARQUET} already exists and is frozen.")
        print("       To re-create deliberately: bump REGISTRY_VERSION or delete the file.")
        sys.exit(1)

    print("=" * 64)
    print("BATCH 2 - BUILD & FREEZE PILOT QUERY SET")
    print("=" * 64)

    print("Loading GEO-Bench (downloads/caches on first run)...")
    ds, prov = load_geobench()
    print(f"  repo            : {prov['geobench_repo']}")
    print(f"  configs found   : {prov['available_configs']}")
    print(f"  resolved config : {prov['resolved_config']}")
    print(f"  resolved split  : {prov['resolved_split']}")
    print(f"  rows in split   : {prov['num_rows']}")
    print(f"  datasets lib    : {prov['datasets_lib_version']}")
    print(f"  dataset version : {prov['dataset_version']}")
    print(f"  dataset license : {prov['dataset_license']}  (verify on the HF dataset card)")
    print(f"  fingerprint     : {prov['fingerprint']}")

    n = prov["num_rows"]
    if not n:
        print("[FAIL] Could not determine row count; aborting.")
        sys.exit(1)

    print("-" * 64)
    print("Field names on record 0:", list(ds[0].keys()))
    print("\nTwo sample records (values truncated to 160 chars):")
    for i in (0, 1):
        row = ds[i]
        preview = {k: (str(v)[:160] + ("..." if len(str(v)) > 160 else "")) for k, v in row.items()}
        print(f"  [{i}] {preview}")
    print("\nTag distribution (top 15):")
    for tag, count in profile_tags(ds, top=15).items():
        print(f"    {count:>6}  {tag}")

    seed = set_global_seeds(cfg.SEED)
    k = cfg.N_QUERIES
    if n < k:
        print(f"[FAIL] only {n} rows available, need {k}.")
        sys.exit(1)
    idx = sorted(random.sample(range(n), k))
    print("-" * 64)
    print(f"Selected {k} rows with seed={seed}: indices {idx}")

    rows, seen = [], {}
    for i in idx:
        r = ds[i]
        qtext = r.get(cfg.QUERY_FIELD, "")
        qid = derive_query_id(qtext)
        if qid in seen and seen[qid] != qtext:
            print(f"[FAIL] query_id collision: {qid} maps to two different queries.")
            print("       Increase QUERY_ID_HASH_LEN in thesis_config and re-run.")
            sys.exit(1)
        seen[qid] = qtext
        rows.append({
            "query_id": qid,
            "query_text": qtext,
            "tags": r.get("tags"),
            "sources": r.get("sources"),
            "sugg_idx": r.get("sugg_idx"),
            "geobench_index": i,
            "geobench_config": prov["resolved_config"],
            "geobench_split": prov["resolved_split"],
        })

    df = pd.DataFrame(rows)

    df.to_parquet(REGISTRY_PARQUET, index=False)
    df.assign(tags=df["tags"].astype(str), sources=df["sources"].astype(str)) \
      .to_csv(REGISTRY_CSV, index=False)

    manifest = {
        "thesis_title": cfg.THESIS_TITLE,
        "artifact": "pilot_query_registry",
        "version": REGISTRY_VERSION,
        "frozen": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_selected": k,
        "selection_method": "seeded_random",
        "seed": seed,
        "query_id_scheme": f"sha256(query_text.strip())[:{cfg.QUERY_ID_HASH_LEN}], prefix '{cfg.QUERY_ID_PREFIX}_'",
        "query_id_hash_len": cfg.QUERY_ID_HASH_LEN,
        "selected_indices": idx,
        "query_ids": [r["query_id"] for r in rows],
        "geobench": prov,
        "registry_file": str(REGISTRY_PARQUET.relative_to(ROOT)),
        "preview_file": str(REGISTRY_CSV.relative_to(ROOT)),
        "notes": [
            "Pilot loads the latest GEO-Bench; pin a commit SHA via revision= for production.",
            "Link-validation of source URLs and full domain/intent profiling are part of full Stage 1 (scale-up).",
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, default=str))

    print("-" * 64)
    print("FROZEN pilot query set:")
    print(f"  registry : {REGISTRY_PARQUET.relative_to(ROOT)}  ({len(df)} rows)")
    print(f"  preview  : {REGISTRY_CSV.relative_to(ROOT)}")
    print(f"  manifest : {MANIFEST_JSON.relative_to(ROOT)}")
    print("\nSelected queries:")
    for r in rows:
        print(f"  {r['query_id']}  | {str(r['query_text'])[:90]}")
    print("=" * 64)
    print("RESULT: PASS - pilot query set built and frozen.")
    print("=" * 64)


if __name__ == "__main__":
    main()
