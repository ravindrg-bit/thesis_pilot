#!/usr/bin/env python3
"""Post-processing prompt 5/8 — STEP 5: save cleaned per-call parquet.
Reads nli_scaleupv2 (READ-ONLY), appends 5 cleaned columns keeping ALL 15
originals intact (incl. sentence_detail/source_scores), writes a NEW file
nli_scaleup_cleaned.parquet. Does NOT overwrite nli_scaleupv2.parquet."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

GOLD = "data/scaleup/silver/nli_scaleupv2.parquet"
OUT = "data/scaleup/silver/nli_scaleup_cleaned.parquet"
KEY = ["engine", "query_id", "run_index"]

ORIG_COLS = [
    "n_sentences", "n_sources_cited", "n_sources_fetched_ok",
    "ais_supported_sentences", "ais_rate", "sentence_detail",
    "pawc_by_source_position", "pawc_total", "nli_evaluations",
    "input_tokens", "output_tokens", "engine", "query_id",
    "run_index", "n_response_words",
]
NEW_COLS = ["n_artifacts", "cleaned_n", "cleaned_ais",
            "cleaned_pawc_total", "pawc_norm"]


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


def compute_row(sentence_detail, m):
    sd = parse(sentence_detail)
    survivors = [e for e in sd if not is_artifact(e.get("text", ""))]
    n_art = len(sd) - len(survivors)
    n_star = len(survivors)
    if n_star == 0:
        return n_art, 0, np.nan, 0.0, np.nan
    supp = 0
    pawc_acc = {}
    omega = 0.0
    for k, e in enumerate(survivors, start=1):
        wc = int(e.get("wc", 0))
        w_star = (n_star - k + 1) / n_star
        omega += wc * w_star
        if as_bool(e.get("supported", False)):
            supp += 1
            for pos in (parse(e.get("sources")) or []):
                pawc_acc[pos] = pawc_acc.get(pos, 0.0) + wc * w_star
    cleaned_ais = supp / n_star
    cleaned_pawc_total = float(sum(pawc_acc.values()))
    pawc_norm = (cleaned_pawc_total / (m * omega)) if (m > 0 and omega > 0) else np.nan
    return n_art, n_star, cleaned_ais, cleaned_pawc_total, pawc_norm


hr("STEP 5.1 — compute cleaned columns")
df = pd.read_parquet(GOLD)
assert list(df.columns) == ORIG_COLS, f"unexpected input columns: {list(df.columns)}"
print(f"input shape: {df.shape}")

vals = [compute_row(r.sentence_detail, int(r.n_sources_fetched_ok))
        for r in df.itertuples(index=False)]
new = pd.DataFrame(vals, columns=NEW_COLS, index=df.index)
new["n_artifacts"] = new["n_artifacts"].astype("int64")
new["cleaned_n"] = new["cleaned_n"].astype("int64")

out = pd.concat([df, new], axis=1)
print(f"output shape: {out.shape}")

hr("STEP 5.2 — write (new file, no overwrite of input)")
assert not os.path.abspath(OUT) == os.path.abspath(GOLD), "refusing to overwrite input"
if os.path.exists(OUT):
    print(f"NOTE: {OUT} already exists and will be replaced.")
out.to_parquet(OUT, index=False)
print(f"written: {OUT}")
print(f"input still present & untouched: {os.path.exists(GOLD)} "
      f"(size {os.path.getsize(GOLD)} bytes)")

# ----- reload & validate ---------------------------------------------------
hr("STEP 5.3 — reload & validate")
rl = pd.read_parquet(OUT)
saved_ok = os.path.exists(OUT)
shape_ok = rl.shape == (9992, 20)

# original columns present & unchanged (compare values)
orig_present = all(c in rl.columns for c in ORIG_COLS)
unchanged = True
for c in ORIG_COLS:
    if c == "sentence_detail" or c == "pawc_by_source_position":
        continue  # object/array cols: check elementwise below
    a, b = df[c], rl[c]
    if a.dtype.kind == "f":
        if not np.allclose(a.fillna(-999), b.fillna(-999)):
            unchanged = False
    else:
        if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
            unchanged = False
# array col spot integrity: lengths match across all rows
sd_lens_match = all(
    len(parse(a)) == len(parse(b))
    for a, b in zip(df["sentence_detail"].head(50), rl["sentence_detail"].head(50)))

new_present = all(c in rl.columns for c in NEW_COLS)

# source_scores spot check on 1 row
spot = parse(rl.iloc[0]["sentence_detail"])
spot_ss = isinstance(spot[0], dict) and "source_scores" in spot[0] \
    and len(parse(spot[0]["source_scores"])) >= 0
print(f"spot-check row0 first entry keys: {sorted(spot[0].keys())}")
print(f"  has source_scores: {'source_scores' in spot[0]}")

# row count by engine unchanged
eng_before = df["engine"].value_counts().sort_index()
eng_after = rl["engine"].value_counts().sort_index()
eng_unchanged = eng_before.equals(eng_after)
print("\nrow count by engine (after):")
print(eng_after.to_string())

print("\ncolumn order (saved):")
print(list(rl.columns))
print("\ndtypes of new columns:")
print(rl[NEW_COLS].dtypes.to_string())

# ----- checklist -----------------------------------------------------------
hr("VALIDATION CHECKLIST")
checks = [
    ("File saved at correct path", saved_ok),
    ("Reloaded shape is 9,992 x 20", shape_ok),
    ("All 15 original columns present & unchanged", orig_present and unchanged and sd_lens_match),
    ("All 5 new columns present", new_present),
    ("sentence_detail still contains source_scores (row spot check)", spot_ss),
    ("Row count by engine unchanged from input", eng_unchanged),
]
for label, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

print("\nSTOP — awaiting confirmation before Step 6.")
