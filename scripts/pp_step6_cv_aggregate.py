#!/usr/bin/env python3
"""Post-processing prompt 6/8 — STEP 6: CV aggregation.
Mirrors pilot cell_aggregates schema exactly (13 cols) + 3 pawc_norm cols.
Groups nli_scaleup_cleaned by (engine, query_id) over the k=8 runs.
Writes a NEW file cell_aggregates_scaleup.parquet."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

CLEANED = "data/scaleup/silver/nli_scaleup_cleaned.parquet"
PILOT_AGG = "data/pilot/gold/cell_aggregates_pilot.parquet"
OUT = "data/scaleup/gold/cell_aggregates_scaleup.parquet"

PILOT_COLS = ["engine", "query_id", "pawc_mean", "pawc_sd", "pawc_cv",
              "ais_mean", "ais_sd", "ais_cv", "n_runs", "n_runs_ais",
              "did_search_rate", "citations_consistent", "n_cited_mean"]
NEW_COLS = ["pawc_norm_mean", "pawc_norm_sd", "pawc_norm_cv"]


def hr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def cv(sd, mean):
    return (sd / mean) if (mean is not None and mean == mean and mean != 0) else np.nan


hr("STEP 6.1 — confirm pilot schema")
pilot = pd.read_parquet(PILOT_AGG)
print("pilot columns:", list(pilot.columns))
assert list(pilot.columns) == PILOT_COLS, "pilot schema drift!"

hr("STEP 6.2 — aggregate scaleup by (engine, query_id)")
df = pd.read_parquet(CLEANED)
print(f"cleaned rows: {len(df)}")


def agg_group(g):
    pawc = g["cleaned_pawc_total"]              # never NaN
    ais = g["cleaned_ais"]                      # NaN where N*=0
    norm = g["pawc_norm"]                       # NaN where M=0
    cited_pos = g["n_sources_cited"] > 0

    pawc_mean = pawc.mean()
    pawc_sd = pawc.std(ddof=1)
    ais_mean = ais.mean()                       # skipna default True
    ais_sd = ais.std(ddof=1)                    # skipna default True
    norm_mean = norm.mean()
    norm_sd = norm.std(ddof=1)

    return pd.Series({
        "pawc_mean": pawc_mean,
        "pawc_sd": pawc_sd,
        "pawc_cv": cv(pawc_sd, pawc_mean),
        "ais_mean": ais_mean,
        "ais_sd": ais_sd,
        "ais_cv": cv(ais_sd, ais_mean),
        "n_runs": int(pawc.notna().sum()),
        "n_runs_ais": int(ais.notna().sum()),
        "did_search_rate": cited_pos.mean(),
        "citations_consistent": bool(cited_pos.nunique() == 1),
        "n_cited_mean": g["n_sources_cited"].mean(),
        "pawc_norm_mean": norm_mean,
        "pawc_norm_sd": norm_sd,
        "pawc_norm_cv": cv(norm_sd, norm_mean),
    })


agg = (df.groupby(["engine", "query_id"], sort=True)
         .apply(agg_group, include_groups=False)
         .reset_index())

# enforce exact column order + pilot dtypes
agg = agg[PILOT_COLS + NEW_COLS]
agg["n_runs"] = agg["n_runs"].astype("int64")
agg["n_runs_ais"] = agg["n_runs_ais"].astype("int64")
agg["citations_consistent"] = agg["citations_consistent"].astype(bool)
print(f"aggregated rows: {len(agg)}")

hr("STEP 6.3 — locate missing cell")
qids = {e: set(g["query_id"]) for e, g in df.groupby("engine")}
ref = qids["chatgpt"]
missing = {e: sorted(ref - q) for e, q in qids.items() if (ref - q)}
print("query_ids present for chatgpt but absent per engine:", missing)

hr("STEP 6.4 — write (new file)")
assert not os.path.exists(OUT) or os.path.abspath(OUT) != os.path.abspath(CLEANED)
agg.to_parquet(OUT, index=False)
print(f"written: {OUT}")

hr("STEP 6.5 — reload & dtypes")
rl = pd.read_parquet(OUT)
print(f"shape: {rl.shape}")
print(rl.dtypes.to_string())

hr("STEP 6.6 — per-engine median CV summary")
cvsum = (rl.groupby("engine")
           .agg(median_ais_cv=("ais_cv", "median"),
                median_pawc_cv=("pawc_cv", "median"),
                median_pawc_norm_cv=("pawc_norm_cv", "median"),
                n_cells=("query_id", "size"))
           .reset_index())
for c in ["median_ais_cv", "median_pawc_cv", "median_pawc_norm_cv"]:
    cvsum[c] = cvsum[c].round(4)
print(cvsum.to_string(index=False))

# ----- checklist -----------------------------------------------------------
hr("VALIDATION CHECKLIST")
row_ok = len(rl) == 1249
miss_ok = missing == {"kimi": ["gb_3fcf760b1a2ea4f8"]}
pilot_cols_ok = all(c in rl.columns for c in PILOT_COLS)
new_cols_ok = all(c in rl.columns for c in NEW_COLS)
total_cols_ok = rl.shape[1] == 16
n_runs_ok = bool((rl["n_runs"] <= 8).all())

checks = [
    ("Exactly 1,249 rows", row_ok),
    ("Missing cell is exactly kimi/gb_3fcf760b1a2ea4f8", miss_ok),
    ("All 13 pilot columns present with matching names", pilot_cols_ok),
    ("3 new columns present (pawc_norm_mean/sd/cv)", new_cols_ok),
    ("Total columns: 16", total_cols_ok),
    ("No n_runs exceeds 8", n_runs_ok),
    ("Per-engine median CV printed", True),
]
for label, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

print("\nSTOP — awaiting confirmation before Step 7.")
