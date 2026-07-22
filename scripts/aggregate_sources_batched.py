"""Batched aggregator — reads JSONs in chunks, writes parquet incrementally."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUN_PROFILE = os.environ.get("THESIS_RUN_PROFILE")
if not RUN_PROFILE:
    sys.stderr.write("[STOP] THESIS_RUN_PROFILE not set\n")
    sys.exit(1)

from src.env import load_env
load_env()
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import thesis_config as cfg

COLS = ["url_canonical","domain","fetch_status","http_status_code",
        "content_length","cleaned_text","fetch_timestamp_utc","title_from_html"]

bronze_dir = cfg.SILVER / "sources_bronze"
out_path = cfg.SILVER / f"sources_{cfg.RUN_PROFILE}.parquet"
files = sorted(bronze_dir.glob("*.json"))
print(f"Files: {len(files)}, batching writes...", flush=True)

BATCH = 2000
writer = None
seen = set()
total = 0

for i in range(0, len(files), BATCH):
    batch = files[i:i+BATCH]
    rows = []
    for f in batch:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["url_canonical"] not in seen:
                seen.add(r["url_canonical"])
                rows.append(r)
        except Exception:
            pass
    if not rows:
        continue
    df = pd.DataFrame(rows)[COLS].astype({"http_status_code":"int64","content_length":"int64"})
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    total += len(df)
    print(f"  batch {i//BATCH + 1}: +{len(df)} rows, total {total}", flush=True)
    del df, table, rows

if writer:
    writer.close()
print(f"[OK] Wrote {total} rows to {out_path}", flush=True)
print(f"File size: {out_path.stat().st_size:,} bytes", flush=True)
