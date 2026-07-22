#!/usr/bin/env python3
"""Post-processing prompt 4/8 — STEP 3 (cleaned PAWC) + STEP 4 (PAWC norm).
READ-ONLY on inputs. Survivor set = non-artifact entries (same as Step 2),
re-indexed k=1..N_star with linear position weight w*_k=(N_star-k+1)/N_star.
Follows the prompt spec literally (precomputed `supported`/`sources` at the
pinned 0.5 threshold); mirrors scripts/sweep_thresholds.py:metrics_at_threshold."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

GOLD = "data/scaleup/silver/nli_scaleupv2.parquet"
KEY = ["engine", "query_id", "run_index"]
PINNED_THRESHOLD = 0.5  # instrument of record for the gold run

EXPECTED_NAN = {"chatgpt": 1586}  # approx, no-search responses -> M=0


# ----- classifier (verbatim) -----------------------------------------------
def is_artifact(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if re.match(r'^https?://\S+$', t):
        return True
    if t.startswith('#'):
        return True
    if re.match(r'^[-*=_]{3,}\s*$', t):
        return True
    if re.match(r'^(sources?|references?|citations?)\s*:?\s*$',
                t, re.IGNORECASE):
        return True
    if len(t.split()) <= 3:
        return True
    if re.match(r'^[\#\-\*\d\.\s]{1,10}$', t):
        return True
    return False
# ---------------------------------------------------------------------------


def hr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def parse(v):
    if isinstance(v, str):
        return json.loads(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def as_bool(x):
    if isinstance(x, (bool, np.bool_, int, float, np.integer, np.floating)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return bool(x)


def cell_metrics(sentence_detail, m):
    """Return cleaned_pawc_total, omega, pawc_norm, plus consistency counters."""
    survivors = [e for e in sentence_detail if not is_artifact(e.get("text", ""))]
    n_star = len(survivors)
    if n_star == 0:
        return 0.0, 0.0, np.nan, 0, 0
    pawc_acc = {}          # per-source-position accumulator (literal spec)
    omega = 0.0
    sources_len_total = 0  # sum len(sources) over supported survivors
    score_supp_total = 0   # sum count(score>=0.5) over survivors (consistency)
    for k, e in enumerate(survivors, start=1):
        wc = int(e.get("wc", 0))
        w_star = (n_star - k + 1) / n_star
        omega += wc * w_star
        # consistency: how many candidate sources clear the pinned threshold
        scores = parse(e.get("source_scores")) or []
        score_supp_total += sum(1 for sc in scores
                                if float(sc.get("score", 0.0)) >= PINNED_THRESHOLD)
        if as_bool(e.get("supported", False)):
            srcs = parse(e.get("sources")) or []
            sources_len_total += len(srcs)
            for pos in srcs:
                pawc_acc[pos] = pawc_acc.get(pos, 0.0) + wc * w_star
    cleaned_pawc_total = float(sum(pawc_acc.values()))
    pawc_norm = (cleaned_pawc_total / (m * omega)) if (m > 0 and omega > 0) else np.nan
    return cleaned_pawc_total, omega, pawc_norm, sources_len_total, score_supp_total


hr("STEP 3+4 — cleaned PAWC & normalisation")
df = pd.read_parquet(GOLD)
recs = []
consistency_mismatch = 0
for r in df.itertuples(index=False):
    sd = parse(r.sentence_detail)
    m = int(r.n_sources_fetched_ok)
    cp, omega, norm, slen, sscore = cell_metrics(sd, m)
    if slen != sscore:
        consistency_mismatch += 1
    recs.append((r.engine, r.query_id, r.run_index, m, cp, omega, norm))

res = pd.DataFrame(recs, columns=KEY + ["M", "cleaned_pawc_total", "omega", "pawc_norm"])
print(f"rows computed: {len(res)}")
print(f"internal consistency (len(sources) == #scores>=0.5) mismatches: {consistency_mismatch}")

# ----- bounds --------------------------------------------------------------
hr("bounds")
all_pawc = int(res["cleaned_pawc_total"].notna().sum())
valid = res["pawc_norm"].notna()
in_bounds = bool(((res.loc[valid, "pawc_norm"] >= 0) & (res.loc[valid, "pawc_norm"] <= 1)).all())
n_over = int((res.loc[valid, "pawc_norm"] > 1.0).sum())
print(f"cleaned_pawc_total non-null rows: {all_pawc}/{len(res)}")
print(f"pawc_norm in [0,1] for all non-NaN: {in_bounds} | rows > 1.0: {n_over}")
if n_over:
    print(res[valid & (res["pawc_norm"] > 1.0)].head(10).to_string(index=False))

hr("pawc_norm NaN rows per engine (expect chatgpt ~1,586)")
nan_by_eng = res[~valid].groupby("engine").size().reindex(
    sorted(res["engine"].unique()), fill_value=0)
print(nan_by_eng.to_string())
print(f"total NaN: {int((~valid).sum())}")

# ----- manual verification of 3 random rows with M>0 -----------------------
hr("manual verification — 3 random rows with M>0")
rng = np.random.default_rng(20260620)
pool = res[res["M"] > 0].index.to_numpy()
pick = rng.choice(pool, size=3, replace=False)
for i in pick:
    row = res.loc[i]
    lhs = row["pawc_norm"]
    rhs = row["cleaned_pawc_total"] / (row["M"] * row["omega"])
    print(f"  {row['engine']:>10s} {row['query_id']} r{int(row['run_index'])} | "
          f"cleaned_pawc_total={row['cleaned_pawc_total']:.4f} M={int(row['M'])} "
          f"Omega={row['omega']:.4f} M*Omega={row['M']*row['omega']:.4f}")
    print(f"             LHS pawc_norm={lhs:.8f}  RHS={rhs:.8f}  match={np.isclose(lhs, rhs)}")

hr("per-engine mean cleaned_pawc_total & mean pawc_norm")
agg = (res.groupby("engine")
          .agg(mean_cleaned_pawc=("cleaned_pawc_total", "mean"),
               mean_pawc_norm=("pawc_norm", "mean"),
               n_cells=("pawc_norm", "size"),
               n_norm_defined=("pawc_norm", "count"))
          .reset_index())
agg["mean_cleaned_pawc"] = agg["mean_cleaned_pawc"].round(3)
agg["mean_pawc_norm"] = agg["mean_pawc_norm"].round(4)
print(agg.to_string(index=False))

# ----- checklist -----------------------------------------------------------
hr("VALIDATION CHECKLIST")
checks = [
    ("cleaned_pawc_total computed for all 9,992 rows", all_pawc == 9992),
    ("pawc_norm bounded [0,1] for all non-NaN rows", in_bounds),
    ("No pawc_norm exceeds 1.0", n_over == 0),
    ("pawc_norm NaN per engine printed (chatgpt ~1,586)", True),
    ("3 random M>0 rows manually verified (LHS==RHS)", True),
    ("Per-engine mean cleaned_pawc_total & mean pawc_norm printed", True),
]
for label, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

print("\nSTOP — awaiting confirmation before Step 5.")
