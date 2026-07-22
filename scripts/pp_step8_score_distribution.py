#!/usr/bin/env python3
"""Post-processing prompt 8/8 — STEP 8: entailment score distribution.
READ-ONLY. Extracts every continuous source_scores value across the dataset,
tagged with artifact-flag and engine; produces Tables D/E/F, a 20-bin
histogram, and threshold-band percentages (Andre Comment 4)."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

CLEANED = "data/scaleup/silver/nli_scaleup_cleaned.parquet"
ENGINES = ["chatgpt", "claude", "gemini", "kimi", "perplexity"]


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
    if re.match(r'^(sources?|references?|citations?)\s*:?\s*$', t, re.IGNORECASE):
        return True
    if len(t.split()) <= 3:
        return True
    if re.match(r'^[\#\-\*\d\.\s]{1,10}$', t):
        return True
    return False


def hr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def parse(v):
    return v.tolist() if isinstance(v, np.ndarray) else v


def pct(a, q):
    return float(np.percentile(a, q))


hr("STEP 8.1 — extract all entailment scores")
df = pd.read_parquet(CLEANED)
scores, artflag, engcode = [], [], []
eng_index = {e: i for i, e in enumerate(ENGINES)}
for r in df.itertuples(index=False):
    ec = eng_index[r.engine]
    for s in parse(r.sentence_detail):
        art = is_artifact(s.get("text", ""))
        for x in (parse(s.get("source_scores")) or []):
            scores.append(float(x["score"]))
            artflag.append(art)
            engcode.append(ec)

scores = np.asarray(scores, dtype=float)
artflag = np.asarray(artflag, dtype=bool)
engcode = np.asarray(engcode, dtype=int)
N = scores.size
print(f"total scores extracted: {N:,}")
print(f"sum(nli_evaluations) over rows: {int(df['nli_evaluations'].sum()):,}")

# ----- TABLE D -------------------------------------------------------------
hr("TABLE D — overall score distribution")
rowD = {
    "count": N, "mean": scores.mean(), "std": scores.std(ddof=1),
    "min": scores.min(), "p10": pct(scores, 10), "p25": pct(scores, 25),
    "p50": pct(scores, 50), "p75": pct(scores, 75), "p90": pct(scores, 90),
    "max": scores.max(),
}
print(" ".join(f"{k}={v:,.4f}" if k != "count" else f"{k}={v:,}"
               for k, v in rowD.items()))

# ----- TABLE E -------------------------------------------------------------
hr("TABLE E — per-engine score distribution")
rowsE = []
for e in ENGINES:
    m = engcode == eng_index[e]
    s = scores[m]
    rowsE.append([e, s.size, s.mean(), np.median(s), s.std(ddof=1),
                  pct(s, 10), pct(s, 90)])
tableE = pd.DataFrame(rowsE, columns=["engine", "count", "mean", "median", "std", "p10", "p90"])
for c in ["mean", "median", "std", "p10", "p90"]:
    tableE[c] = tableE[c].round(4)
print(tableE.to_string(index=False))
print(f"per-engine counts sum: {int(tableE['count'].sum()):,}")

# ----- TABLE F -------------------------------------------------------------
hr("TABLE F — score distribution: artifact vs non-artifact")
rowsF = []
for label, mask in [("artifact", artflag), ("non_artifact", ~artflag)]:
    s = scores[mask]
    rowsF.append([label, s.size, s.mean(), np.median(s), s.std(ddof=1)])
tableF = pd.DataFrame(rowsF, columns=["segment_type", "count", "mean", "median", "std"])
for c in ["mean", "median", "std"]:
    tableF[c] = tableF[c].round(4)
print(tableF.to_string(index=False))

# ----- histogram -----------------------------------------------------------
hr("HISTOGRAM — 20 bins over [0,1] (Andre Comment 4)")
counts, edges = np.histogram(scores, bins=20, range=(0.0, 1.0))
hist_pct = counts / N * 100
peak = counts.max()
for i in range(20):
    bar = "#" * int(round(hist_pct[i] / 100 * 60))
    print(f"  [{edges[i]:.2f},{edges[i+1]:.2f})  {counts[i]:>8,}  {hist_pct[i]:6.2f}%  {bar}")
print(f"  histogram total: {int(counts.sum()):,} ({hist_pct.sum():.2f}%)")

# ----- threshold bands -----------------------------------------------------
hr("threshold bands (% of all scores)")
b_lt30 = (scores < 0.30).mean() * 100
b_30_50 = ((scores >= 0.30) & (scores < 0.50)).mean() * 100
b_50_70 = ((scores >= 0.50) & (scores < 0.70)).mean() * 100
b_gt70 = (scores >= 0.70).mean() * 100
print(f"  < 0.30        : {b_lt30:6.2f}%")
print(f"  [0.30, 0.50)  : {b_30_50:6.2f}%")
print(f"  [0.50, 0.70)  : {b_50_70:6.2f}%")
print(f"  >= 0.70       : {b_gt70:6.2f}%")
print(f"  band total    : {b_lt30 + b_30_50 + b_50_70 + b_gt70:6.2f}%")

# ----- checklist -----------------------------------------------------------
hr("VALIDATION CHECKLIST")
total_pos = N > 0
range_ok = bool((scores >= 0.0).all() and (scores <= 1.0).all())
count_match = N == int(df["nli_evaluations"].sum())
eng_sum_ok = int(tableE["count"].sum()) == N
hist_sum_ok = abs(hist_pct.sum() - 100.0) < 1e-6
checks = [
    ("Total scores extracted > 0", total_pos),
    ("Score range is [0,1] (no negatives / nothing > 1)", range_ok),
    ("Score count matches sum of nli_evaluations", count_match),
    ("Per-engine counts sum to total", eng_sum_ok),
    ("Histogram bins sum to 100%", hist_sum_ok),
]
for label, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

print("\nPOST-PROCESSING PIPELINE COMPLETE.")
