"""
run_nli_pilot.py - FULL-PILOT NLI attribution driver (Option 3).

Derived from scripts/smoke_nli.py, but DATA-DRIVEN: it processes every cell present in the
silver responses table - all distinct run_index values, every (query_id, engine) row - so the
same script runs unchanged on smoke, pilot, or scaleup silver (loop bounds come from the data,
never hardcoded). Writes one row per cell to cfg.GOLD / nli_{RUN_PROFILE}.parquet,
KEEPING pawc_by_source_position for downstream per-source PAWC normalisation.

PAWC + AIS come from the SAME formulas as the smoke; only the scope (full silver) differs.
Run:  python scripts/run_nli_pilot.py

CHECKPOINT/RESUME (I/O only — metric math untouched):
  Completed cells are persisted to nli_{RUN_PROFILE}.partial.parquet every NLI_CKPT_EVERY
  cells (default 200) and at the end of every run. On restart, cells already present in
  the partial are skipped, so a crash / pod shutdown costs at most NLI_CKPT_EVERY cells
  of GPU work instead of the whole run. The final parquet is written once at the end
  (unchanged behaviour), then the partial is removed.

USE_PREFETCHED_SOURCES (default True when sources parquet exists):
  When the sources_{profile}.parquet built by build_sources.py is present, pre-populate
  the in-memory fetch cache before scoring begins. fetch_source_text() checks the cache
  first, so every citation URL is served from the parquet — no live HTTP requests during
  NLI scoring, making attribution fully reproducible from immutable artefacts.
  Set USE_PREFETCHED_SOURCES=False (or delete the sources parquet) to fall back to live
  fetching (original behaviour).
"""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.env import load_env
load_env()
import pandas as pd
import thesis_config as cfg
from src import nli_attribution as nli
from src.silver import canonicalise_url

responses = pd.read_parquet(cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet")
citations = pd.read_parquet(cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet")

# Scope: exclude engines whose collection is incomplete. Driven from cfg.COLLECTION_ENGINES
# so the exclusion list is maintained in one place (thesis_config._EXCLUDED_ENGINES).
responses = responses[responses.engine.isin(cfg.COLLECTION_ENGINES)].reset_index(drop=True)

# Loop bounds come from the DATA, not hardcoded constants.
queries = sorted(responses.query_id.unique())
engines = sorted(responses.engine.unique())
runs = sorted(responses.run_index.unique())
total = len(responses)

# --- Pre-populate fetch cache from sources parquet (reproducible attribution) ---
USE_PREFETCHED_SOURCES = True
sources_path = cfg.SILVER / f"sources_{cfg.RUN_PROFILE}.parquet"

cache: dict = {}
if USE_PREFETCHED_SOURCES and sources_path.exists():
    # The parquet stores FULL page text. The original live-fetch path truncated every
    # source to cfg.JUDGE_SOURCE_CHAR_LIMIT inside fetch_source_text BEFORE the NLI step
    # chunked it. A cache HIT returns early and bypasses that truncation, so we re-apply
    # the SAME window here when loading the cache. This (a) reproduces the original NLI
    # scores exactly — the point of prefetching is provenance, not changing the metric —
    # and (b) avoids chunking multi-MB pages. To score against full text instead, raise
    # or remove this limit (the full content is preserved in the parquet either way).
    #
    # STREAMED, not pd.read_parquet of the whole file: cleaned_text uncompressed is
    # ~6.4 GB at scaleup and a whole-frame pandas load peaks at SEVERAL TIMES that
    # (verified OOM-kill on a 16 GB machine; pilot's 775 MB parquet peaked at 4.65 GB
    # RSS). Only the truncated windows are kept, so peak memory ≈ cache + one batch.
    import pyarrow.parquet as pq
    _src_window = cfg.JUDGE_SOURCE_CHAR_LIMIT
    n_ok, n_fail = 0, 0
    _pf = pq.ParquetFile(sources_path)
    for _batch in _pf.iter_batches(batch_size=256,
                                   columns=["url_canonical", "fetch_status", "cleaned_text"]):
        for _url, _status, _text in zip(_batch.column(0).to_pylist(),
                                        _batch.column(1).to_pylist(),
                                        _batch.column(2).to_pylist()):
            key = canonicalise_url(_url)
            if _status == "ok" and _text:
                cache[key] = (_text[:_src_window], "ok")
                n_ok += 1
            else:
                # failed / empty -> (None, status); fetch_source_text returns it, NLI skips it
                cache[key] = (None, _status)
                n_fail += 1
    print(f"[prefetch] loaded {n_ok} ok + {n_fail} failed URLs from {sources_path.name} "
          f"(source window {_src_window} chars)")

    # Phase 3 req 4: warn on any citation URL absent from the sources parquet. A miss would
    # silently fall through to a LIVE fetch in fetch_source_text, breaking reproducibility.
    cited_keys = {canonicalise_url(u) for u in citations["url"].dropna().unique()}
    missing = cited_keys - set(cache.keys())
    if missing:
        print(f"[prefetch][WARN] {len(missing)} cited URL(s) missing from sources parquet "
              f"-> these will be LIVE-fetched. Re-run build_sources.py to capture them.")
        for u in list(missing)[:10]:
            print(f"    missing: {u}")
    else:
        print(f"[prefetch] all {len(cited_keys)} cited URLs present in sources parquet (0 live fetches)")
elif USE_PREFETCHED_SOURCES:
    print(f"[prefetch] sources parquet not found at {sources_path} — falling back to live fetch")

# --- Checkpoint/resume (crash-safe gold; I/O only, metric math untouched) ---------------
CKPT_EVERY = int(os.environ.get("NLI_CKPT_EVERY", "200"))
# NLI per-response tables are silver-layer entity tables (one row per response). Per the
# 2026-06-26 medallion realignment (README §1), scaleup writes them to silver; the frozen
# pilot keeps its historical gold location.
nli_dir = cfg.SILVER if cfg.RUN_PROFILE == "scaleup" else cfg.GOLD
out_path = nli_dir / f"nli_{cfg.RUN_PROFILE}.parquet"
ckpt_path = nli_dir / f"nli_{cfg.RUN_PROFILE}.partial.parquet"

all_rows = []
done = set()
if ckpt_path.exists():
    _prev = pd.read_parquet(ckpt_path)
    all_rows = _prev.to_dict("records")
    for _r in all_rows:   # checkpoint stores the two variable-schema cols as JSON strings
        _r["pawc_by_source_position"] = json.loads(_r["pawc_by_source_position"])
        _r["fetch_status"] = json.loads(_r["fetch_status"])
    done = {(r["engine"], r["query_id"], int(r["run_index"])) for r in all_rows}
    print(f"[resume] {len(all_rows)} completed cells loaded from {ckpt_path.name} — skipping them")


def _write_checkpoint():
    """Atomic cumulative checkpoint: tmp + os.replace (same pattern as build_sources.py).
    pawc_by_source_position and fetch_status are JSON-ENCODED in the checkpoint: pyarrow
    cannot write an all-empty struct column, and a leading block of zero-citation cells
    (e.g. chatgpt, 75.8% zero-cite in scaleup) would otherwise crash the first checkpoint.
    The FINAL parquet keeps the original struct types — resume decodes back to dicts."""
    nli_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in all_rows:
        c = dict(r)
        c["pawc_by_source_position"] = json.dumps(c["pawc_by_source_position"])
        c["fetch_status"] = json.dumps(c["fetch_status"])
        rows.append(c)
    tmp = ckpt_path.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, ckpt_path)


t0 = time.time()
print("=" * 72)
print(f"NLI ATTRIBUTION PILOT | {len(queries)} q x {len(engines)} engines x {len(runs)} runs = {total} cells")
print(f"model: {cfg.NLI_MODEL}  |  threshold: {cfg.NLI_ENTAILMENT_THRESHOLD}  |  device: {cfg.NLI_DEVICE}")
print(f"sentence segmenter: {cfg.SENTENCE_SEGMENTER}  (pinned — instrument of record, see thesis_config)")
print(f"source mode: {'prefetched parquet' if (USE_PREFETCHED_SOURCES and sources_path.exists()) else 'live fetch'}")
print(f"checkpoint: every {CKPT_EVERY} cells + end of each run -> {ckpt_path.name}")
print("=" * 72)

for run_idx in runs:
    run_idx = int(run_idx)
    run_rows = []
    sub = responses[responses.run_index == run_idx]
    for _, cell in sub.iterrows():
        eng = cell["engine"]
        qid = cell["query_id"]
        if (eng, qid, run_idx) in done:        # resume: already in the checkpoint
            continue
        answer = cell["answer_text"]
        srcs = (citations[(citations.engine == eng) & (citations.query_id == qid)
                          & (citations.run_index == run_idx)]
                .sort_values("position")[["position", "url"]].to_dict("records"))
        res = nli.attribute_cell_nli(answer, srcs, cache)
        res.update({"engine": eng, "query_id": qid, "run_index": run_idx,
                    "n_response_words": len((answer or "").split())})
        run_rows.append(res)
        all_rows.append(res)
        if len(all_rows) % CKPT_EVERY == 0:
            _write_checkpoint()
        ais = res["ais_rate"]
        print(f"r{run_idx} {eng:11s} {qid[:18]} | sents {res['n_sentences']:3d} | "
              f"src {res['n_sources_fetched_ok']}/{res['n_sources_cited']} | "
              f"AIS {round(ais, 2) if ais is not None else None} | "
              f"PAWC {res['pawc_total']} | NLI evals {res['nli_evaluations']}")

    # checkpoint at end of every run, then per-engine summary of NEWLY computed cells
    _write_checkpoint()
    if run_rows:
        rdf = pd.DataFrame(run_rows)
        print(f"\n--- run r{run_idx} complete: {len(rdf)} new cells | elapsed {(time.time()-t0)/60:.1f} min ---")
        summ = (rdf.groupby("engine").agg(
                    cells=("query_id", "count"),
                    mean_ais=("ais_rate", "mean"),
                    mean_pawc=("pawc_total", "mean"),
                    mean_sents=("n_sentences", "mean")).round(2))
        print(summ.to_string())
        print()
    else:
        print(f"\n--- run r{run_idx}: all cells already in checkpoint (resumed) ---\n")

# ---- write full gold: KEEP pawc_by_source_position; drop only fetch_status ----
df = pd.DataFrame(all_rows)
cfg.GOLD.mkdir(parents=True, exist_ok=True)
df.drop(columns=["fetch_status"]).to_parquet(out_path, index=False)
ckpt_path.unlink(missing_ok=True)   # final write succeeded -> checkpoint no longer needed

elapsed = time.time() - t0
print("=" * 72)
print(f"PILOT NLI COMPLETE | {len(df)} cells | {int(df['nli_evaluations'].sum())} NLI evals | "
      f"{elapsed/60:.1f} min ({elapsed/max(1, len(df)):.1f} s/cell)")
print(f"written: {out_path}")
print("=" * 72)
