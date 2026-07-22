#!/usr/bin/env python3
"""Post-processing prompt 1/8 — STEP 0: load & inspect scaleup NLI gold v2.
READ-ONLY. Does not modify any input files."""
import json
import numpy as np
import pandas as pd


def _len(x):
    """Length of a list/ndarray/None without ambiguous truthiness."""
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return 0


def _as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

GOLD = "data/scaleup/silver/nli_scaleupv2.parquet"
PILOT_AGG = "data/pilot/gold/cell_aggregates_pilot.parquet"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------- STEP 0.1
hr("STEP 0.1 — LOAD gold v2")
df = pd.read_parquet(GOLD)
print(f"shape: {df.shape}")

print("\ncolumns & dtypes:")
print(df.dtypes.to_string())

print("\nengine x run_index crosstab:")
ct = pd.crosstab(df["engine"], df["run_index"])
print(ct.to_string())
print("\nrow totals per engine:")
print(df["engine"].value_counts().sort_index().to_string())

print("\nnulls per column:")
print(df.isnull().sum().to_string())

key = ["engine", "query_id", "run_index"]
n_dupe = int(df.duplicated(subset=key).sum())
print(f"\nduplicate (engine, query_id, run_index) rows: {n_dupe}")

# ---------------------------------------------------------------- STEP 0.2
hr("STEP 0.2 — sentence_detail / source_scores")
ok_col = "n_sources_fetched_ok"
cand = df[df[ok_col] > 0] if ok_col in df.columns else df.iloc[0:0]
if len(cand) == 0:
    print(f"WARNING: no rows with {ok_col} > 0; falling back to any non-empty sentence_detail")


def parse_sd(v):
    if isinstance(v, str):
        return json.loads(v)
    return _as_list(v)


scores_collected = []
chosen_row = None

for idx, row in cand.iterrows():
    sd = parse_sd(row["sentence_detail"])
    if _len(sd) == 0:
        continue
    if chosen_row is None:
        chosen_row = (idx, row)
    for sent in sd:
        if not isinstance(sent, dict):
            continue
        ss = _as_list(sent.get("source_scores"))
        if _len(ss):
            scores_collected.extend(
                [s.get("score") for s in ss if isinstance(s, dict) and "score" in s]
            )
    if chosen_row is not None and len(scores_collected) > 50:
        break

if chosen_row is not None:
    idx, row = chosen_row
    sd = parse_sd(row["sentence_detail"])
    print(f"chosen row idx={idx} | engine={row['engine']} | query_id={row['query_id']} "
          f"| run_index={row['run_index']} | {ok_col}={row.get(ok_col)}")
    first = sd[0]
    print("\nkeys in first sentence_detail entry:")
    print(sorted(first.keys()) if isinstance(first, dict) else type(first))

    # find first sentence that actually has source_scores
    fss = None
    for sent in sd:
        ss = _as_list(sent.get("source_scores")) if isinstance(sent, dict) else None
        if _len(ss):
            fss = ss[0]
            break
    print("\nfirst source_scores entry:")
    print(fss)
    print("type of pos:", type(fss.get("pos")).__name__ if isinstance(fss, dict) else "NA",
          "| type of score:", type(fss.get("score")).__name__ if isinstance(fss, dict) else "NA")

arr = np.array([s for s in scores_collected if s is not None], dtype=float)
n_unique = len(np.unique(arr)) if arr.size else 0
non_binary = bool(arr.size and not np.all(np.isin(arr, [0.0, 1.0])))
print(f"\nsource_scores collected: {arr.size} | unique values: {n_unique} | "
      f"min={arr.min() if arr.size else 'NA'} max={arr.max() if arr.size else 'NA'}")
print(f"continuous (not just 0/1): {non_binary}")
if arr.size:
    print("sample values:", np.round(arr[:10], 4).tolist())

# ---------------------------------------------------------------- STEP 0.3
hr("STEP 0.3 — pilot cell_aggregates schema")
pilot_read_ok = True
try:
    pa = pd.read_parquet(PILOT_AGG)
    print(f"shape: {pa.shape}")
    print("\ncolumns & dtypes:")
    print(pa.dtypes.to_string())
except Exception as e:
    pilot_read_ok = False
    print(f"FAILED to read pilot aggregates: {e}")

# ---------------------------------------------------------------- CHECKLIST
hr("VALIDATION CHECKLIST")
engines = sorted(df["engine"].unique().tolist())
expected_engines = ["chatgpt", "claude", "gemini", "kimi", "perplexity"]

# per-run counts
kimi_ok = others_ok = True
if "kimi" in ct.index:
    kimi_ok = bool((ct.loc["kimi"] == 249).all())
for e in ct.index:
    if e == "kimi":
        continue
    if not bool((ct.loc[e] == 250).all()):
        others_ok = False

sd_nonnull = bool(df["sentence_detail"].notnull().all())
ss_present = bool(arr.size > 0)

checks = [
    ("Exactly 9,992 rows", df.shape[0] == 9992),
    ("5 engines: chatgpt, claude, gemini, kimi, perplexity", engines == expected_engines),
    ("Kimi 249/run, others 250/run", kimi_ok and others_ok),
    ("Zero duplicate keys", n_dupe == 0),
    ("sentence_detail present & non-null for all rows", sd_nonnull),
    ("source_scores present in sentence_detail entries", ss_present),
    ("Scores are continuous floats (not binary)", non_binary),
    ("Pilot aggregates schema read successfully", pilot_read_ok),
]
for label, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

print("\nengines found:", engines)
print("STOP — awaiting confirmation before Step 1.")
