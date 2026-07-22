#!/usr/bin/env python3
"""
Worked example — cleaned AIS, cleaned PAWC, and PAWC_Norm for ONE query.

  Notation:
    N       total sentences in the response                 (n_sentences)
    N*      surviving sentences after artifact removal       (cleaned_n)
    k       1-based position AMONG SURVIVORS ONLY (re-indexed after dropping artifacts)
    w*_k    position weight of survivor k = (N* - k + 1) / N*   (linear decay)
    l_k     word count of survivor k                          (wc)
    e(s_k,c) DeBERTa-MNLI max-window entailment prob, survivor k vs. cited source c
    tau     entailment threshold, pinned at 0.5
    C       set of cited sources; M = |C| = sources fetched ok (n_sources_fetched_ok)
    Omega   theoretical-max position-weighted word mass = sum_{k=1..N*} l_k * w*_k

  Formulas:
    cleaned_ais        = |{k : exists c in C, e(s_k,c) >= tau}| / N*
    PAWC(c)            = sum_{k : e(s_k,c) >= tau} l_k * w*_k
    cleaned_pawc_total = sum_{c in C} PAWC(c)
    pawc_norm          = cleaned_pawc_total / (M * Omega)

Source of truth: src/attribution.py:1-14,75-78, scripts/pp_step3_4_cleaned_pawc.py:68-93,
scripts/pp_step5_save_cleaned.py:75-96, README.md Sec 5.4 and Sec 19.

Read-only: loads the gold parquet, recomputes by hand, and asserts the recomputed
values match what the pipeline already stored. Run: python scripts/worked_example_cleaned_metrics.py
"""
import json
import re

import numpy as np
import pandas as pd

GOLD = "data/scaleup/silver/nli_scaleup_cleaned.parquet"
TAU = 0.5


# ----- artifact classifier (verbatim from scripts/pp_step5_save_cleaned.py) -----
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


def hr(t=""):
    print("\n" + "=" * 92)
    if t:
        print(t)
        print("=" * 92)


def print_formula_header():
    hr("FORMULAS AND NOTATION")
    print("""
  N       total sentences in the response                 (n_sentences)
  N*      surviving sentences after artifact removal       (cleaned_n)
  k       1-based position AMONG SURVIVORS ONLY (re-indexed after dropping artifacts)
  w*_k    position weight of survivor k = (N* - k + 1) / N*   (linear decay)
  l_k     word count of survivor k                          (wc)
  e(s_k,c) entailment probability, survivor k vs. cited source c (DeBERTa-MNLI, max window)
  tau     entailment threshold, pinned at 0.5
  C       set of cited sources; M = |C| = sources fetched ok (n_sources_fetched_ok)
  Omega   theoretical-max position-weighted word mass = sum_{k=1..N*} l_k * w*_k

  cleaned_ais        = |{k : exists c in C, e(s_k,c) >= tau}| / N*
  PAWC(c)            = sum_{k : e(s_k,c) >= tau} l_k * w*_k
  cleaned_pawc_total = sum_{c in C} PAWC(c)
  pawc_norm          = cleaned_pawc_total / (M * Omega)
""")


def pick_example(df):
    """Pick one illustrative response: >=1 artifact dropped, M>1 sources, AIS strictly between 0 and 1."""
    cand = df[(df["n_artifacts"] > 0) & (df["n_sources_fetched_ok"] > 1) &
              (df["cleaned_ais"] > 0) & (df["cleaned_ais"] < 1)]
    row = cand.iloc[0]
    print(f"Selected: engine={row.engine!r} query_id={row.query_id!r} run_index={row.run_index} "
          f"(chosen for: n_artifacts={row.n_artifacts} > 0, "
          f"n_sources_fetched_ok={row.n_sources_fetched_ok} > 1, "
          f"0 < cleaned_ais={row.cleaned_ais:.4f} < 1)")
    return row


def worked_example(row):
    sd_all = parse(row.sentence_detail)
    n = len(sd_all)
    m = int(row.n_sources_fetched_ok)

    hr(f"WORKED EXAMPLE  |  {row.engine}__{row.query_id}__r{row.run_index}")
    print(f"N (total sentences) = {n}   M (sources fetched ok) = {m}   tau = {TAU}")

    survivors = [e for e in sd_all if not is_artifact(e.get("text", ""))]
    n_star = len(survivors)
    n_art = n - n_star
    print(f"N* (survivors after dropping {n_art} artifact sentence(s)) = {n_star}")

    print("\n-- dropped artifact sentences (original 1-based idx, excluded from k-indexing) --")
    for e in sd_all:
        if is_artifact(e.get("text", "")):
            print(f"  idx={int(e['idx']):>2}  {str(e['text'])[:60]!r}")

    hr()
    print(f"{'k':>3} {'orig_idx':>8} {'l_k':>4} {'w*_k':>7} {'supp?':>6} {'sources':>10} "
          f"{'term=l_k*w*_k':>14}   survivor text (clipped)")
    omega = 0.0
    supported = 0
    pawc_acc = {}
    for k, e in enumerate(survivors, start=1):
        l_k = int(e.get("wc", 0))
        w_star = (n_star - k + 1) / n_star
        term = l_k * w_star
        omega += term
        supp = as_bool(e.get("supported", False))
        srcs = parse(e.get("sources")) or []
        if supp:
            supported += 1
            for pos in srcs:
                pawc_acc[pos] = pawc_acc.get(pos, 0.0) + term
        print(f"{k:>3} {int(e['idx']):>8} {l_k:>4} {w_star:>7.4f} {str(supp):>6} "
              f"{str(srcs):>10} {term:>14.4f}   {str(e['text'])[:40]!r}")

    cleaned_ais = supported / n_star
    cleaned_pawc_total = float(sum(pawc_acc.values()))
    pawc_norm = cleaned_pawc_total / (m * omega) if (m > 0 and omega > 0) else np.nan

    hr("PER-SOURCE PAWC(c) BREAKDOWN")
    if pawc_acc:
        for pos, val in sorted(pawc_acc.items()):
            print(f"  source position {pos}: PAWC(c) = {val:.4f}")
    else:
        print("  (no source received any supported-sentence credit)")

    hr("RECOMPUTED vs. STORED")
    print(f"Omega (theoretical max mass)      = {omega:.4f}")
    print(f"cleaned_ais        recomputed     = {supported}/{n_star} = {cleaned_ais:.6f}   "
          f"stored = {row.cleaned_ais:.6f}")
    print(f"cleaned_pawc_total recomputed     = {cleaned_pawc_total:.6f}   "
          f"stored = {row.cleaned_pawc_total:.6f}")
    print(f"pawc_norm          recomputed     = {pawc_norm:.6f}   "
          f"stored = {row.pawc_norm:.6f}")

    assert abs(cleaned_ais - row.cleaned_ais) < 1e-9, "cleaned_ais mismatch"
    assert abs(cleaned_pawc_total - row.cleaned_pawc_total) < 1e-6, "cleaned_pawc_total mismatch"
    assert abs(pawc_norm - row.pawc_norm) < 1e-6, "pawc_norm mismatch"
    print("\nPASS - hand-recomputed cleaned_ais, cleaned_pawc_total, and pawc_norm "
          "match the stored gold values.")


if __name__ == "__main__":
    print_formula_header()
    df = pd.read_parquet(GOLD)
    example_row = pick_example(df)
    worked_example(example_row)
