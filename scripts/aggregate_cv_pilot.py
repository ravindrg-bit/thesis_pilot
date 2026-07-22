"""
aggregate_cv_pilot.py — CV aggregation across the k=3 repeats (per-call -> per-cell).

Collapses the per-call gold parquet (1,500 rows = 100 queries x 5 engines x 3 runs)
into a second gold-layer table at the (query x engine) CELL level (500 rows). For each
cell we summarise the 3 repeats with a mean, an SD, and the coefficient of variation
(CV = SD / mean) — the repeated-measures determinism control from V3 sec.7.2 (Atil et
al. 2025; Yuan et al. 2025; Schulte et al. 2026), since temperature/seed are not
settable on this model generation.

Pure pandas: load -> groupby(engine, query_id) -> agg + arithmetic -> write. It does
NOT touch the per-call parquet and adds nothing to src/. Read the gold path from
thesis_config (cfg.GOLD resolves to data/<run_profile>/gold), never hardcoded.

Run:  python scripts/aggregate_cv_pilot.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import thesis_config as cfg

# Per-call gold (input) and per-cell gold (output) both live under cfg.GOLD — the
# profile-aware gold dir (data/pilot/gold for the pilot). Path from config, not hardcoded.
PER_CALL_GOLD = cfg.GOLD / f"nli_{cfg.RUN_PROFILE}.parquet"
CELL_GOLD_OUT = cfg.GOLD / f"cell_aggregates_{cfg.RUN_PROFILE}.parquet"


def main():
    df = pd.read_parquet(PER_CALL_GOLD)

    # --- did_search is INFERRED, not logged: a call "searched" iff it returned >=1 cited
    #     source. We never recorded the raw web_search_call flag in gold, so n_sources_cited>0
    #     is the proxy (a search that yields no usable citation reads as no-search here). ---
    df["did_search"] = df["n_sources_cited"] > 0

    # The lone null-AIS row (Gemini / gb_57e198d356ad745f / run 1, n_sentences=0) is KEPT for
    # PAWC (all 3 repeats) and excluded from AIS only: pandas mean/std/count skip NaN, so AIS
    # for that one cell aggregates over its 2 non-null repeats while PAWC still uses all 3.
    g = df.groupby(["engine", "query_id"], as_index=False).agg(
        pawc_mean=("pawc_total", "mean"),
        pawc_sd=("pawc_total", "std"),            # pandas default ddof=1
        n_runs=("pawc_total", "size"),            # PAWC repeats: all rows in the cell (== 3)
        ais_mean=("ais_rate", "mean"),            # skips the null AIS row
        ais_sd=("ais_rate", "std"),               # skips null; ddof=1
        n_runs_ais=("ais_rate", "count"),         # non-null AIS repeats (<3 only at the null cell)
        did_search_rate=("did_search", "mean"),   # fraction of runs that searched: 0/.33/.67/1.0
        _did_search_nunique=("did_search", "nunique"),
        n_cited_mean=("n_sources_cited", "mean"),
    )

    # CV = SD / mean (the ONLY definition used). Zero-mean cells -> CV = NaN: not an error to
    # hide but SIGNAL — the engine returned nothing useful on every repeat (e.g. ChatGPT cells
    # that never searched -> pawc_mean=0). These rows MUST survive downstream, NOT be dropped;
    # we null only their CV. Replacing the 0 mean with NaN before dividing avoids 0/0 warnings.
    g["pawc_cv"] = g["pawc_sd"] / g["pawc_mean"].replace(0, np.nan)
    g["ais_cv"] = g["ais_sd"] / g["ais_mean"].replace(0, np.nan)

    # did_search identical across all 3 runs? (nunique==1 over the cell's did_search values)
    g["citations_consistent"] = g.pop("_did_search_nunique").eq(1)

    cell = g[[
        "engine", "query_id",
        "pawc_mean", "pawc_sd", "pawc_cv",
        "ais_mean", "ais_sd", "ais_cv",
        "n_runs", "n_runs_ais",
        "did_search_rate", "citations_consistent", "n_cited_mean",
    ]]

    # --- write + verify shape (tolerant of documented collection gaps) -------------------
    # Expected cells = distinct (engine, query) pairs actually COLLECTED in silver, after the
    # same engine exclusion run_nli_pilot applied — NOT the theoretical N×engines grid. Scaleup
    # gap: kimi/gb_3fcf760b1a2ea4f8 missing all 8 runs -> 1,249 cells, not 1,250
    # (see docs/decision_logs/2026-06-12_kimi_missing_cell.md).
    resp = pd.read_parquet(cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet")
    resp = resp[resp.engine.isin(cfg.COLLECTION_ENGINES)]
    expected_cells = resp.groupby(["engine", "query_id"]).ngroups
    n_active_engines = cell["engine"].nunique()
    theoretical_max = cfg.N_QUERIES * n_active_engines
    assert len(cell) == expected_cells, (
        f"aggregation changed cell count: silver has {expected_cells} (engine,query) cells, "
        f"output has {len(cell)}"
    )
    assert (cell["n_runs"] <= cfg.K).all(), (
        f"a cell aggregated more than {cfg.K} repeats (duplicate per-call rows?)"
    )
    if expected_cells != theoretical_max:
        print(f"[note] {theoretical_max - expected_cells} cell(s) absent vs theoretical "
              f"{theoretical_max} ({cfg.N_QUERIES} q × {n_active_engines} eng) — documented collection gap")
    CELL_GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    cell.to_parquet(CELL_GOLD_OUT, index=False)

    # --- diagnostics --------------------------------------------------------------------
    n_zero_pawc = int(cell["pawc_cv"].isna().sum())        # null pawc_cv <=> zero-PAWC cell
    summary = (
        cell.assign(zero_pawc=cell["pawc_mean"].eq(0))
        .groupby("engine")
        .agg(
            mean_pawc_cv=("pawc_cv", "mean"),              # mean skips null-CV (zero-PAWC) cells
            mean_ais_cv=("ais_cv", "mean"),
            mean_did_search_rate=("did_search_rate", "mean"),
            n_cells_zero_pawc=("zero_pawc", "sum"),
        )
        .round(3)
    )

    print("=" * 72)
    print(f"CV AGGREGATION (k={cfg.K} repeats) | per-call -> per-cell")
    print(f"in : {PER_CALL_GOLD.relative_to(ROOT)}  ({len(df)} rows)")
    print(f"out: {CELL_GOLD_OUT.relative_to(ROOT)}  ({len(cell)} rows)")
    print("=" * 72)
    print(f"Total cells written              : {len(cell)} (expect 500)")
    print(f"Cells with pawc_cv null          : {n_zero_pawc} (zero-PAWC cells)")
    print(f"Cells with ais_cv null           : {int(cell['ais_cv'].isna().sum())}")
    print(f"Cells with did_search_rate < 1.0 : {int((cell['did_search_rate'] < 1.0).sum())} (partial-search cells)")
    print(f"Cells with citations_consistent=False : {int((~cell['citations_consistent']).sum())}")
    print("-" * 72)
    print("Per-engine summary:")
    print(summary.to_string())
    print("=" * 72)


if __name__ == "__main__":
    main()
