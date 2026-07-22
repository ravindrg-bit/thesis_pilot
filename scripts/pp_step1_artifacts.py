#!/usr/bin/env python3
"""Post-processing prompt 2/8 — STEP 1: artifact classification.
READ-ONLY on inputs. Builds an in-memory (engine, query_id, run_index) table
of n_segments_total / n_artifacts and validates against nli_scaleupv2."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.attribution import segment_sentences

SILVER = "data/scaleup/silver/responses_scaleup.parquet"
GOLD = "data/scaleup/silver/nli_scaleupv2.parquet"

KEY = ["engine", "query_id", "run_index"]

# expected per-engine artifact rates (fraction) and tolerance
EXPECTED = {
    "kimi": 0.51,
    "chatgpt": 0.14,
    "claude": 0.15,
    "gemini": 0.13,
    "perplexity": 0.02,
}
TOL_PP = 8.0  # percentage points


# ----- classifier (copied verbatim from prompt, do not modify) -------------
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


hr("STEP 1.1 — segment & classify silver responses")
silver = pd.read_parquet(SILVER)
print(f"silver rows: {len(silver)}")

rows = []
for r in silver.itertuples(index=False):
    segs = segment_sentences(r.answer_text)
    n_total = len(segs)
    n_art = sum(1 for s in segs if is_artifact(s))
    rows.append((r.engine, r.query_id, r.run_index, n_total, n_art))

art = pd.DataFrame(rows, columns=KEY + ["n_segments_total", "n_artifacts"])
print(f"artifact table rows: {len(art)}")

hr("STEP 1.2 — join to nli_scaleupv2")
gold = pd.read_parquet(GOLD)[KEY + ["n_sentences"]]
merged = art.merge(gold, on=KEY, how="outer", indicator=True)
orphans_left = int((merged["_merge"] == "left_only").sum())
orphans_right = int((merged["_merge"] == "right_only").sum())
n_matched = int((merged["_merge"] == "both").sum())
print(f"matched (both): {n_matched} | silver-only orphans: {orphans_left} | "
      f"gold-only orphans: {orphans_right}")

both = merged[merged["_merge"] == "both"].copy()
n_mismatch = int((both["n_segments_total"] != both["n_sentences"]).sum())
print(f"n_segments_total != n_sentences mismatches: {n_mismatch}")
if n_mismatch:
    ex = both[both["n_segments_total"] != both["n_sentences"]].head(10)
    print("sample mismatches:")
    print(ex[KEY + ["n_segments_total", "n_sentences"]].to_string(index=False))

hr("STEP 1.3 — per-engine artifact rates")
agg = (art.groupby("engine")
          .agg(n_segments_total=("n_segments_total", "sum"),
               n_artifacts=("n_artifacts", "sum"))
          .reset_index())
agg["artifact_rate"] = agg["n_artifacts"] / agg["n_segments_total"]
agg["expected"] = agg["engine"].map(EXPECTED)
agg["dev_pp"] = (agg["artifact_rate"] - agg["expected"]).abs() * 100
agg["within_8pp"] = agg["dev_pp"] <= TOL_PP
disp = agg.copy()
disp["artifact_rate"] = (disp["artifact_rate"] * 100).round(2).astype(str) + "%"
disp["expected"] = (disp["expected"] * 100).round(0).astype(int).astype(str) + "%"
disp["dev_pp"] = disp["dev_pp"].round(2)
print(disp.to_string(index=False))

# ----- checklist -----------------------------------------------------------
hr("VALIDATION CHECKLIST")
all_processed = len(art) == 9992
join_ok = (n_matched == 9992) and (orphans_left == 0) and (orphans_right == 0)
seg_match = (n_mismatch == 0)
rates_printed = True
within_tol = bool(agg["within_8pp"].all())

checks = [
    ("All 9,992 response rows processed", all_processed),
    ("Join to nli_scaleupv2: all 9,992 matched, zero orphans", join_ok),
    ("n_segments_total == n_sentences for all rows (zero mismatches)", seg_match),
    ("Per-engine artifact rates printed", rates_printed),
    ("No engine deviates >8pp from expected", within_tol),
]
for label, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

print("\nSTOP — awaiting confirmation before Step 2.")
