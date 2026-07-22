#!/usr/bin/env python3
"""build_v2_premise_cache.py — max-entailing 200-token window per V2 sampled pair.

The gold NLI stage scores each (sentence, source) pair by chunking the FULL source into
overlapping 200-token windows (stride 50) and keeping the MAX entailment across windows
(src/nli_attribution.py). Only that max score is persisted, not the winning window. To show
a human annotator the exact passage DeBERTa judged, this script re-runs the model on the
deterministic 150-pair V2 sample and recovers, per pair, the window that produced the score.

Efficiency: windows are scored in order of lexical overlap with the sentence and scanning
stops once a window reaches the stored max (early-exit) — exhaustive scoring is infeasible
(some sources exceed 2M tokens). `premise_matched` flags whether the recovered window
reproduced the stored score within 0.02; unmatched pairs (rare, only very long sources whose
winning window is a low-overlap passage beyond the cap) are reported.

Sampling MUST match validation.ipynb cell 9 EXACTLY: gold-layer is_artifact() eligibility +
degenerate-drop + seed 20260606 + 30/engine. READ-ONLY on data/. Writes
analysis/v2_maxwindow_premises.parquet keyed by (engine, query_id, run_index,
sentence_index, source_position). Regenerate with:
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/build_v2_premise_cache.py
"""
import logging
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

import thesis_config as cfg
from flag_segment_artifacts import is_artifact          # gold-layer filter (population parity)
from src import nli_attribution as M                    # exact chunking / model config

SEED, NPER = 20260606, 30
ENG = ["chatgpt", "claude", "gemini", "kimi", "perplexity"]
# The gold NLI premise is the source truncated to JUDGE_SOURCE_CHAR_LIMIT (4,000) chars BEFORE
# windowing — fetch_source_text() returns txt[:char_limit] (src/attribution.py). Matching this cut
# is what reproduces the stored scores (and bounds each source to ~6-7 windows).
CHAR_CAP = cfg.JUDGE_SOURCE_CHAR_LIMIT
WIN_CAP = 200        # window cap (never binds after the 4,000-char cut; kept as a safety bound)
BATCH = 16           # NLI batch size (kept modest to bound MPS memory)
CLEANED = os.path.join(ROOT, "data/scaleup/silver/nli_scaleup_cleaned.parquet")
CIT = os.path.join(ROOT, "data/scaleup/silver/citations_scaleup.parquet")
SRC = os.path.join(ROOT, "data/scaleup/silver/sources_scaleup.parquet")
OUT = os.path.join(ROOT, "analysis/v2_maxwindow_premises.parquet")

_WORD = re.compile(r"[a-z0-9]+")


def _mps_empty():
    if cfg.NLI_DEVICE == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _strip_md(t):
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t or "")
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[#*_`>~\-\|]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_degenerate(text):
    """Worksheet-quality drop (on top of the gold-layer filter): a segment too short or
    structural to adjudicate. Reconcile-with-gold caveat: the gold cleaned-AIS keeps these."""
    raw = (text or "").strip()
    if re.match(r"^\(?\s*\[[^\]]+\]\([^)]+\)\s*\)?[\s\-#*.>|]*$", raw):   # only a markdown link
        return True
    if re.match(r"^\(?\s*https?://\S+\s*\)?[\s\-#*.>|]*$", raw):          # only a URL
        return True
    if raw.startswith("#"):                                              # heading-only
        return True
    if len(_strip_md(raw).split()) < 6:                                  # < 6 words after markdown
        return True
    return False


def build_sample():
    nli = pd.read_parquet(CLEANED, columns=["engine", "query_id", "run_index", "sentence_detail"])
    rows = []
    for r in nli.itertuples(index=False):
        for s in r.sentence_detail:
            if is_artifact(s["text"]):                 # gold-layer eligibility (AIS parity)
                continue
            for sc in s["source_scores"]:
                rows.append((r.engine, r.query_id, int(r.run_index), int(s["idx"]),
                             s["text"], int(sc["pos"]), float(sc["score"])))
    p = pd.DataFrame(rows, columns=["engine", "query_id", "run_index", "sentence_index",
                                    "sentence_text", "source_position", "deberta_score"])
    cit = (pd.read_parquet(CIT, columns=["engine", "query_id", "run_index", "position",
                                         "url_canonical", "domain"])
             .rename(columns={"position": "source_position", "domain": "source_domain"}))
    p = p.merge(cit, on=["engine", "query_id", "run_index", "source_position"], how="left")
    meta = pd.read_parquet(SRC, columns=["url_canonical", "fetch_status"])
    ok = set(meta.loc[meta["fetch_status"].eq("ok"), "url_canonical"])
    p = p[p["url_canonical"].isin(ok)].copy()
    pool = p["engine"].value_counts().reindex(ENG)
    p["degen"] = p["sentence_text"].map(is_degenerate)
    degen = p[p["degen"]]["engine"].value_counts().reindex(ENG).fillna(0).astype(int)
    clean = p[~p["degen"]].copy()
    samp = pd.concat([clean[clean.engine == e].sample(n=NPER, random_state=SEED) for e in ENG],
                     ignore_index=True)
    return samp, pool, degen, clean["engine"].value_counts().reindex(ENG)


def fetch_text(urls):
    want, txt = set(urls), {}
    for b in pq.ParquetFile(SRC).iter_batches(batch_size=400,
                                              columns=["url_canonical", "cleaned_text"]):
        for u, c in zip(b.column("url_canonical").to_pylist(),
                        b.column("cleaned_text").to_pylist()):
            if u in want and u not in txt:
                txt[u] = c or ""
        if len(txt) >= len(want):
            break
    return txt


def make_windows(text, tok):
    """Overlapping 200/50-token windows (mirrors src/nli_attribution._chunk_source)."""
    ids = tok((text or "")[:CHAR_CAP], add_special_tokens=False, truncation=False)["input_ids"]
    ct, st = cfg.NLI_SOURCE_CHUNK_TOKENS, cfg.NLI_SOURCE_CHUNK_STRIDE
    if len(ids) <= ct:
        return [tok.decode(ids, skip_special_tokens=True)] if ids else [""]
    out, i = [], 0
    while i < len(ids):
        out.append(tok.decode(ids[i:i + ct], skip_special_tokens=True))
        if i + ct >= len(ids):
            break
        i += (ct - st)
    return out


def max_window(sentence, text, tok, model, eidx, stored, bs=BATCH):
    wins = make_windows(text, tok)
    sw = set(_WORD.findall(sentence.lower()))
    order = sorted(range(len(wins)),
                   key=lambda k: len(sw & set(_WORD.findall(wins[k].lower()))), reverse=True)[:WIN_CAP]
    best, best_win, target = -1.0, (wins[order[0]] if order else ""), stored - 0.01
    for i in range(0, len(order), bs):
        batch = [wins[k] for k in order[i:i + bs]]
        enc = tok(batch, [sentence] * len(batch), return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(cfg.NLI_DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(**enc).logits, dim=-1)[:, eidx].cpu().tolist()
        del enc
        _mps_empty()
        for w, pp in zip(batch, probs):
            if pp > best:
                best, best_win = pp, w
        if best >= target:
            break
    return best, best_win, len(wins)


def main():
    t0 = time.time()
    samp, pool, degen, clean = build_sample()
    print("eligible pool (gold-layer is_artifact + ok source) per engine:\n", pool.to_string())
    print("\ndegenerate pairs dropped from worksheet per engine:\n", degen.to_string())
    print("\nclean pool per engine:\n", clean.to_string())
    print(f"\nsample: {len(samp)} pairs, {samp['url_canonical'].nunique()} unique urls")

    txt = fetch_text(samp["url_canonical"].tolist())
    M._load_nli()
    tok, model, eidx = M._NLI["tok"], M._NLI["model"], M._NLI["entail_idx"]
    print(f"NLI model loaded on {cfg.NLI_DEVICE}\n")

    KEYS = ["engine", "query_id", "run_index", "sentence_index", "source_position"]
    PARTIAL = OUT + ".partial"
    done = {}
    if os.path.exists(PARTIAL):                                    # resume after a crash
        for r in pd.read_parquet(PARTIAL).itertuples(index=False):
            done[(r.engine, r.query_id, int(r.run_index), int(r.sentence_index), int(r.source_position))] = (
                r.source_premise_maxwindow, float(r.premise_recomputed_score),
                int(r.premise_n_tokens), bool(r.premise_matched))
        print(f"resuming: {len(done)} pairs already checkpointed in {os.path.relpath(PARTIAL, ROOT)}\n")

    recs = []
    for j, r in enumerate(samp.itertuples(index=False), 1):
        key = (r.engine, r.query_id, int(r.run_index), int(r.sentence_index), int(r.source_position))
        if key in done:
            w, b, nt, mt = done[key]
        else:
            b, w, _ = max_window(r.sentence_text, txt.get(r.url_canonical, ""), tok, model, eidx, r.deberta_score)
            b = round(float(b), 5)
            nt = len(tok(w, add_special_tokens=True)["input_ids"])
            mt = bool(abs(b - r.deberta_score) <= 0.02)
            done[key] = (w, b, nt, mt)
            _mps_empty()
        recs.append(dict(engine=r.engine, query_id=r.query_id, run_index=int(r.run_index),
                         sentence_index=int(r.sentence_index), source_position=int(r.source_position),
                         source_premise_maxwindow=w, premise_recomputed_score=b,
                         premise_n_tokens=nt, premise_matched=mt))
        if j % 10 == 0 or j == len(samp):
            pd.DataFrame(recs).to_parquet(PARTIAL, index=False)      # crash-safe checkpoint
            print(f"  {j}/{len(samp)} pairs  ({time.time()-t0:.0f}s elapsed)", flush=True)

    out = samp.merge(pd.DataFrame(recs), on=KEYS, how="left")
    keep = ["engine", "query_id", "run_index", "sentence_index", "source_position", "source_domain",
            "sentence_text", "deberta_score", "source_premise_maxwindow",
            "premise_recomputed_score", "premise_n_tokens", "premise_matched"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[keep].to_parquet(OUT, index=False)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)

    ntok, matched = out["premise_n_tokens"].tolist(), out["premise_matched"].tolist()
    print(f"\nwrote {OUT} ({len(out)} rows) in {time.time()-t0:.0f}s")
    print(f"premise match rate (|recomputed - stored| <= 0.02): "
          f"{np.mean(matched)*100:.1f}%  ({int(np.sum(matched))}/{len(matched)})")
    print(f"premise token length: max {max(ntok)}  (all <= 512: {max(ntok) <= 512})")
    um = out[~out["premise_matched"]]
    if len(um):
        print("unmatched pairs (premise = best window found within cap):")
        for r in um.itertuples(index=False):
            print(f"  {r.engine:10s} stored={r.deberta_score:.3f} recomputed={r.premise_recomputed_score:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
