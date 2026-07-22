"""
flag_segment_artifacts.py — segment-level non-claim artifact flagging.

No pre-segmented sentence table exists in this project: the NLI pipeline segments
answer_text on the fly via src.attribution.segment_sentences() (pinned regex backend,
cfg.SENTENCE_SEGMENTER) and never persists the sentences. This script reconstructs the
segment-level table from data/scaleup/silver/responses_scaleup.parquet using that SAME
function (so segment boundaries / counts match the measurement instrument), flags each
sentence with is_artifact(), reports the artifact rate per engine, and — only if the
per-engine rates land within tolerance of the expected ground-truth numbers — writes
analysis/sentences_with_artifact_flag.parquet.

Reads only the silver parquet; never modifies any original file.
"""

import os
import re
import sys

import pandas as pd

# Force the pinned instrument-of-record segmenter (default already "regex"); set BEFORE
# importing thesis_config so the module-level read picks it up.
os.environ.setdefault("SENTENCE_SEGMENTER", "regex")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import thesis_config as cfg                       # noqa: E402
from src.attribution import segment_sentences     # noqa: E402

assert cfg.SENTENCE_SEGMENTER == "regex", (
    f"expected regex segmenter, got {cfg.SENTENCE_SEGMENTER!r}"
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PARQUET = os.path.join(ROOT, "data/scaleup/silver/responses_scaleup.parquet")
OUT_PARQUET = os.path.join(ROOT, "analysis/sentences_with_artifact_flag.parquet")

# Expected ground-truth artifact rates (fractions) and the flag tolerance.
EXPECTED = {"kimi": 0.47, "chatgpt": 0.19, "claude": 0.14, "gemini": 0.14, "perplexity": 0.04}
TOL_PP = 5.0   # percentage points


# ----------------------------------------------------------------------------- is_artifact
_URL_RE     = re.compile(r"^https?://\S+$")
_HR_RE      = re.compile(r"^([-*=_])\1{2,}$")              # 3+ of a single -, *, =, _ char, nothing else
_SOURCES_RE = re.compile(r"^(sources|references|citations)\s*:?\s*$", re.IGNORECASE)
_LABEL_RE   = re.compile(r"^[\#\-\*\d\.\s]{1,10}$")        # pure punctuation / number label


def is_artifact(text: str) -> bool:
    """True if `text` is a non-claim artifact (heading, rule, URL, label, stub, …)."""
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
    if len(s.split()) <= 3:         # 3 words or fewer ("Sure.", "Great question!", labels)
        return True
    if _LABEL_RE.match(s):          # pure punctuation / number label
        return True
    return False


# ----------------------------------------------------------------------------- pipeline
def build_segments(responses: pd.DataFrame) -> pd.DataFrame:
    """Explode one-response-per-row into one-sentence-per-row via segment_sentences()."""
    keep = ["query_id", "engine", "run_index", "model_served", "run_profile"]
    rows = []
    for r in responses.itertuples(index=False):
        sents = segment_sentences(r.answer_text)
        for i, sent in enumerate(sents, start=1):
            rows.append({
                "query_id":     r.query_id,
                "engine":       r.engine,
                "run_index":    r.run_index,
                "model_served": r.model_served,
                "run_profile":  r.run_profile,
                "sentence_index": i,          # 1-based, matches position_weight convention
                "n_sentences":  len(sents),
                "sentence":     sent,
            })
    return pd.DataFrame(rows, columns=keep + ["sentence_index", "n_sentences", "sentence"])


def main() -> int:
    responses = pd.read_parquet(SRC_PARQUET)
    print(f"loaded {len(responses):,} responses from {os.path.relpath(SRC_PARQUET, ROOT)}")

    seg = build_segments(responses)
    seg["is_artifact"] = seg["sentence"].apply(is_artifact)
    print(f"segmented into {len(seg):,} sentences "
          f"({seg['is_artifact'].mean()*100:.1f}% artifacts overall)\n")

    # --- artifact rate by engine vs expected -------------------------------------------
    rate = seg.groupby("engine")["is_artifact"].mean().sort_values(ascending=False)
    print("artifact rate by engine:")
    print(f"  {'engine':12s} {'n_sent':>8s} {'rate':>7s} {'expected':>9s} {'delta_pp':>9s}  flag")
    off = []
    for engine, r in rate.items():
        exp = EXPECTED.get(engine)
        n = int((seg["engine"] == engine).sum())
        if exp is None:
            print(f"  {engine:12s} {n:8,d} {r*100:6.1f}% {'   n/a':>9s} {'   n/a':>9s}")
            continue
        delta = (r - exp) * 100
        flag = "  <-- OFF >5pp" if abs(delta) > TOL_PP else ""
        if abs(delta) > TOL_PP:
            off.append(engine)
        print(f"  {engine:12s} {n:8,d} {r*100:6.1f}% {exp*100:8.1f}% {delta:+8.1f}{flag}")

    # --- diagnostics for any off engine ------------------------------------------------
    for engine in off:
        print(f"\n!! {engine} is off by >{TOL_PP}pp — 10 example artifact=True rows:")
        ex = seg[(seg["engine"] == engine) & (seg["is_artifact"])].head(10)
        for j, row in enumerate(ex.itertuples(index=False), 1):
            print(f"  [{j:2d}] {row.sentence[:140]!r}")

    # --- save only if all rates within tolerance ---------------------------------------
    if off:
        print(f"\nNOT saving: {len(off)} engine(s) off by >{TOL_PP}pp ({', '.join(off)}). "
              f"Review the examples above and adjust before saving.")
        return 1

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    seg.to_parquet(OUT_PARQUET, index=False)
    print(f"\nall engines within {TOL_PP}pp -> saved {len(seg):,} rows to "
          f"{os.path.relpath(OUT_PARQUET, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
