#!/usr/bin/env python
"""
sweep_thresholds.py — post-hoc NLI entailment-threshold sensitivity sweep.

The gold NLI run decides support at a single declared threshold (tau=0.50). Because
src/nli_attribution.py now stores the *continuous* per-(sentence, source) entailment
probability in sentence_detail[*].source_scores (one {pos, score} per fetched source),
the support decision can be re-derived at other thresholds WITHOUT a GPU re-run.

For each tau in {0.5, 0.6, 0.7, 0.8, 0.9} this script walks every cell's sentence_detail,
re-derives each sentence's supporting sources as
    [s["pos"] for s in sent["source_scores"] if s["score"] >= tau],
and recomputes the cleaned metrics using the SAME artifact-filtering, survivor
re-indexing, and Omega-normalisation logic as the post-processing step (README §19):

  * cleaned_ais        = supported survivors / N*               (NaN if N* = 0)
  * cleaned_pawc_total = sum_k wc_k * w*_k * (#supporting sources) over survivors k=1..N*
                         with w*_k = (N* - k + 1) / N*
  * pawc_norm          = cleaned_pawc_total / (M * Omega)       (NaN if M = 0 or Omega = 0)
                         M = n_sources_fetched_ok, Omega = sum_k wc_k * w*_k over survivors

It then groups by engine and prints  engine | threshold | mean_cleaned_ais |
mean_cleaned_pawc | mean_pawc_norm  (means skip NaN; per-engine n shown).

Gold parquets produced BEFORE source_scores was added (the existing nli_pilot.parquet /
nli_scaleup.parquet) store only the binary support flag — this script detects the missing
field and prints a clear re-run message instead of misleading numbers.

Read-only: never writes or modifies any file under data/.
"""

import argparse
import math
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import thesis_config as cfg                                # noqa: E402

# Prefer the committed flagging rules; fall back to an identical local copy if the
# sibling module can't be imported (e.g. a non-regex segmenter trips its module assert).
try:
    from flag_segment_artifacts import is_artifact         # noqa: E402
except Exception:                                          # pragma: no cover
    import re

    _URL_RE     = re.compile(r"^https?://\S+$")
    _HR_RE      = re.compile(r"^([-*=_])\1{2,}$")
    _SOURCES_RE = re.compile(r"^(sources|references|citations)\s*:?\s*$", re.IGNORECASE)
    _LABEL_RE   = re.compile(r"^[\#\-\*\d\.\s]{1,10}$")

    def is_artifact(text: str) -> bool:
        """True if `text` is a non-claim artifact (rules mirror README §19)."""
        if text is None:
            return True
        s = text.strip()
        if not s:                       # empty / whitespace only
            return True
        if _URL_RE.match(s):            # bare URL
            return True
        if s.startswith("#"):           # markdown heading
            return True
        if _HR_RE.match(s):             # horizontal rule
            return True
        if _SOURCES_RE.match(s):        # Sources / References / Citations header
            return True
        if len(s.split()) <= 3:         # 3 words or fewer
            return True
        if _LABEL_RE.match(s):          # pure punctuation / number label
            return True
        return False


THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _struct_field_names(t):
    """Field names of a (possibly list-wrapped) pyarrow struct type, else []."""
    while (pa.types.is_list(t) or pa.types.is_large_list(t)
           or pa.types.is_fixed_size_list(t)):
        t = t.value_type
    if not pa.types.is_struct(t):
        return []
    return [t.field(i).name for i in range(t.num_fields)]


def sentence_detail_has_source_scores(path):
    """True iff the parquet's sentence_detail struct carries a source_scores field."""
    schema = pq.read_schema(path)
    if "sentence_detail" not in schema.names:
        return False
    return "source_scores" in _struct_field_names(schema.field("sentence_detail").type)


def survivors_of(sentence_detail):
    """Drop artifact segments; return [(wc, [score, ...]), ...] for survivors, in order.

    Artifact filtering is threshold-independent, so survivors (and their word counts /
    raw entailment scores) are computed once per cell and reused across all thresholds.
    """
    out = []
    for s in sentence_detail:
        if is_artifact(s["text"]):
            continue
        scores = [float(sc["score"]) for sc in s["source_scores"]]
        out.append((int(s["wc"]), scores))
    return out


def metrics_at_threshold(survivors, m, threshold):
    """(cleaned_ais, cleaned_pawc_total, pawc_norm) for one cell at one threshold.

    Survivors are re-indexed k = 1..N*; w*_k = (N* - k + 1) / N* (README §19). A source
    supports a survivor when its continuous score >= threshold; AIS counts survivors with
    >=1 supporter, PAWC credits every supporter full weight, pawc_norm divides by M*Omega.
    """
    n_star = len(survivors)
    if n_star == 0:
        return math.nan, 0.0, math.nan
    supported = 0
    pawc_total = 0.0
    omega = 0.0
    for k, (wc, scores) in enumerate(survivors, start=1):
        w_star = (n_star - k + 1) / n_star
        omega += wc * w_star
        n_support = sum(1 for sc in scores if sc >= threshold)
        if n_support:
            supported += 1
        pawc_total += wc * w_star * n_support
    cleaned_ais = supported / n_star
    pawc_norm = (pawc_total / (m * omega)) if (m > 0 and omega > 0) else math.nan
    return cleaned_ais, pawc_total, pawc_norm


def build_summary(df):
    """Per-(engine, threshold) mean cleaned metrics. Returns a tidy DataFrame."""
    records = []
    for row in df.itertuples(index=False):
        survivors = survivors_of(row.sentence_detail)
        m = int(row.n_sources_fetched_ok)
        for thr in THRESHOLDS:
            ais, pawc, norm = metrics_at_threshold(survivors, m, thr)
            records.append({
                "engine": row.engine,
                "threshold": thr,
                "cleaned_ais": ais,
                "cleaned_pawc_total": pawc,
                "pawc_norm": norm,
            })
    res = pd.DataFrame.from_records(records)
    summary = (
        res.groupby(["engine", "threshold"])
        .agg(
            mean_cleaned_ais=("cleaned_ais", "mean"),       # pandas mean() skips NaN
            mean_cleaned_pawc=("cleaned_pawc_total", "mean"),
            mean_pawc_norm=("pawc_norm", "mean"),
            n_cells=("cleaned_ais", "size"),                # all rows for the engine
            n_pawc_norm=("pawc_norm", "count"),             # cells with M>0 (norm defined)
        )
        .reset_index()
        .sort_values(["engine", "threshold"])
    )
    return summary


def print_summary(summary):
    hdr = (f"{'engine':12s} {'threshold':>9s} {'mean_cleaned_ais':>17s} "
           f"{'mean_cleaned_pawc':>18s} {'mean_pawc_norm':>15s} "
           f"{'n_cells':>8s} {'n_norm':>7s}")
    print(hdr)
    print("-" * len(hdr))
    last_engine = None
    for r in summary.itertuples(index=False):
        if last_engine is not None and r.engine != last_engine:
            print()
        last_engine = r.engine
        ais = "nan" if math.isnan(r.mean_cleaned_ais) else f"{r.mean_cleaned_ais:.4f}"
        norm = "nan" if math.isnan(r.mean_pawc_norm) else f"{r.mean_pawc_norm:.4f}"
        print(f"{r.engine:12s} {r.threshold:9.2f} {ais:>17s} "
              f"{r.mean_cleaned_pawc:18.3f} {norm:>15s} "
              f"{int(r.n_cells):8d} {int(r.n_pawc_norm):7d}")


def main():
    default_path = str(cfg.GOLD / f"nli_{cfg.RUN_PROFILE}.parquet")
    ap = argparse.ArgumentParser(
        description="Post-hoc NLI entailment-threshold sensitivity sweep (README §19).")
    ap.add_argument("parquet", nargs="?", default=default_path,
                    help=f"gold NLI parquet (default: {default_path})")
    args = ap.parse_args()
    path = args.parquet

    if not os.path.exists(path):
        print(f"ERROR: parquet not found: {path}", file=sys.stderr)
        return 1

    if not sentence_detail_has_source_scores(path):
        print(
            "ERROR: this gold parquet has no `source_scores` in sentence_detail, so the\n"
            "       NLI entailment threshold cannot be re-swept post-hoc.\n\n"
            f"  file: {path}\n\n"
            "  It was produced BEFORE continuous entailment scores were stored — its\n"
            "  sentence_detail carries only the binary `supported` flag decided at tau=0.50,\n"
            "  not the per-(sentence, source) entailment probabilities a sweep needs.\n\n"
            "  Fix: re-run `python scripts/run_nli_pilot.py` with the updated\n"
            "  src/nli_attribution.py (which appends source_scores) to produce a\n"
            "  sweep-capable gold parquet, then re-run this script against it.\n"
            "  (The existing nli_pilot.parquet / nli_scaleup.parquet predate the change.)",
            file=sys.stderr,
        )
        return 1

    df = pd.read_parquet(
        path, columns=["engine", "n_sources_fetched_ok", "sentence_detail"])
    print(f"gold: {path}")
    print(f"rows: {len(df):,}   engines: {df['engine'].nunique()}   "
          f"thresholds: {THRESHOLDS}\n")

    summary = build_summary(df)
    print_summary(summary)
    print("\nnote: means skip NaN cells. n_cells = rows per engine; "
          "n_norm = cells with M=n_sources_fetched_ok>0 (pawc_norm defined).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
