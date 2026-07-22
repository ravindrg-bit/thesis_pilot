#!/usr/bin/env python3
"""Post-processing prompt 3/8 — STEP 2: cleaned AIS.
READ-ONLY on inputs. Computes per-row cleaned AIS from the artifact-filtered
survivor set ONLY (never from ais_supported_sentences), plus the
n_artifacts_supported measurement finding."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

GOLD = "data/scaleup/silver/nli_scaleupv2.parquet"
KEY = ["engine", "query_id", "run_index"]

EXPECTED_ART_SUPP = {  # approx, for sanity only
    "kimi": 22500, "claude": 5500, "gemini": 4800,
    "chatgpt": 700, "perplexity": 100,
}


# ----- classifier (copied verbatim, do not modify) -------------------------
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


def parse_sd(v):
    if isinstance(v, str):
        return json.loads(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def as_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    return bool(x)


hr("STEP 2.1 — compute cleaned AIS per row")
df = pd.read_parquet(GOLD)

recs = []
for r in df.itertuples(index=False):
    sd = parse_sd(r.sentence_detail)
    n_sd = len(sd)
    n_art = 0
    n_art_supp = 0
    survivors_supported = 0
    n_star = 0
    for e in sd:
        txt = e.get("text", "")
        sup = as_bool(e.get("supported", False))
        if is_artifact(txt):
            n_art += 1
            if sup:
                n_art_supp += 1
        else:
            n_star += 1
            if sup:
                survivors_supported += 1
    cleaned_ais = (survivors_supported / n_star) if n_star > 0 else np.nan
    recs.append((r.engine, r.query_id, r.run_index, r.n_sentences,
                 n_sd, n_art, n_star, survivors_supported,
                 cleaned_ais, n_art_supp, r.ais_rate))

cols = KEY + ["n_sentences", "n_sd", "n_artifacts", "N_star",
              "cleaned_supported", "cleaned_ais", "n_artifacts_supported",
              "raw_ais_rate"]
res = pd.DataFrame(recs, columns=cols)
print(f"rows computed: {len(res)}")

# ----- integrity checks ----------------------------------------------------
hr("STEP 2.2 — integrity")
sd_len_match = int((res["n_sd"] == res["n_sentences"]).sum())
print(f"len(sentence_detail) == n_sentences: {sd_len_match}/{len(res)}")

mism = res["N_star"] != (res["n_sentences"] - res["n_artifacts"])
n_mism = int(mism.sum())
print(f"N_star != n_sentences - n_artifacts mismatches: {n_mism}")

valid = res["cleaned_ais"].notna()
in_bounds = bool(((res.loc[valid, "cleaned_ais"] >= 0) &
                  (res.loc[valid, "cleaned_ais"] <= 1)).all())
n_over_1 = int((res.loc[valid, "cleaned_ais"] > 1.0).sum())
print(f"cleaned_ais in [0,1] for all non-NaN: {in_bounds} | rows > 1.0: {n_over_1}")
print(f"NaN cleaned_ais rows: {int((~valid).sum())}")

hr("STEP 2.3 — N_star <= 0 per engine (expect ~6, all Gemini)")
zero = res[res["N_star"] <= 0]
if len(zero):
    print(zero["engine"].value_counts().sort_index().to_string())
else:
    print("none")
print(f"total N_star<=0 rows: {len(zero)}")

hr("STEP 2.4 — n_artifacts_supported per engine (measurement finding)")
art_supp = (res.groupby("engine")["n_artifacts_supported"].sum()
              .reset_index().rename(columns={"n_artifacts_supported": "n_art_supp"}))
art_supp["expected_approx"] = art_supp["engine"].map(EXPECTED_ART_SUPP)
print(art_supp.to_string(index=False))

hr("STEP 2.5 — mean cleaned_ais vs mean raw ais_rate per engine")
means = (res.groupby("engine")
            .agg(mean_cleaned_ais=("cleaned_ais", "mean"),
                 mean_raw_ais=("raw_ais_rate", "mean"),
                 n_rows=("cleaned_ais", "size"),
                 n_valid_cleaned=("cleaned_ais", "count"))
            .reset_index())
means["mean_cleaned_ais"] = means["mean_cleaned_ais"].round(4)
means["mean_raw_ais"] = means["mean_raw_ais"].round(4)
means["delta"] = (means["mean_cleaned_ais"] - means["mean_raw_ais"]).round(4)
print(means.to_string(index=False))

# ----- checklist -----------------------------------------------------------
hr("VALIDATION CHECKLIST")
checks = [
    ("N_star == n_sentences - n_artifacts for all 9,992 rows", n_mism == 0),
    ("cleaned_ais bounded [0,1] for all non-NaN rows", in_bounds),
    ("No cleaned_ais exceeds 1.0 (v1 bug must not recur)", n_over_1 == 0),
    ("N_star<=0 counted per engine (printed above)", True),
    ("Per-engine n_artifacts_supported printed", True),
    ("Per-engine mean cleaned_ais vs raw ais_rate printed", True),
]
for label, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

print("\nSTOP — awaiting confirmation before Step 3.")
