"""Build the page-feature modeling frames for the page-features analysis
(notebooks/scaleup/page_features.ipynb).

Three steps, each resumable (skips work whose output already exists; --force rebuilds):

  1. url_features        — stream data/scaleup/silver/sources_scaleup.parquet (3.5 GB)
                           and compute per-URL text features from cleaned_text/title.
  2. citation_events     — one row per citation event from the gold star schema
                           (flat_table1/2/3/4), with position outcomes and entailment
                           outcomes computed over NON-ARTIFACT sentences only.
  3. model frames        — join events x url features x enrichment taxonomy x query
                           facets; winsorize + z-standardize features (pooled);
                           emit event-level and stickiness frames.

Outputs (all under analysis/page_features/):
  url_features.parquet, citation_events.parquet, url_engine_agg.parquet,
  model_frame_events.parquet, model_frame_sticky.parquet,
  model_frame_sticky_domain.parquet

Scaleup-profile only; reads silver/gold/enrichment, never writes to data/.
Model fitting lives in the notebook, not here.

Note: gemini's url_canonical values are per-response vertexaisearch
grounding-redirect tokens (unique per citation), so gemini is excluded from the
page-level stickiness frame; its real source domain (from grounding metadata)
still supports the domain-level frame.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER = os.path.join(ROOT, "data/scaleup/silver")
GOLD = os.path.join(ROOT, "data/scaleup/gold")
ENRICH = os.path.join(ROOT, "data/scaleup/enrichment")
QUERIES = os.path.join(ROOT, "data/scaleup/queries")
OUT_DIR = os.path.join(ROOT, "analysis/page_features")

CAP_CHARS = 30_000   # detailed regex features computed on this text prefix
LEAD_WORDS = 150     # "lead" = first 150 words drawn from substantive lines
TAU_SUP, TAU_CON = 0.50, 0.30  # entailment bands (verbatim from rq3 Q13)

RE_YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
RE_MONTH = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b")
RE_NUMTOK = re.compile(r"\d")
RE_BULLET = re.compile(r"^\s*(?:[-*•·–—]|\d{1,2}[.)])\s+\S")
RE_LISTICLE = re.compile(r"\b(?:top|best)\s+\d+\b|\b\d+\s+(?:ways|tips|things|steps|reasons|examples|ideas|best)\b", re.I)
RE_SENT_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
RE_WORD = re.compile(r"[A-Za-z]+")
STOP = set("the a an and or of to in for on with is are was were be been at by from as it this that".split())

CONT_FEATURES = ["log1p_n_words", "flesch_c", "avg_word_len", "digits_per_100w",
                 "dates_per_100w", "lead_num_token_share", "caps_mid_share", "ttr_1000",
                 "short_line_share", "bullet_line_share", "newlines_per_100w",
                 "question_lines_per_100", "nav_noise_share", "title_len_words",
                 "title_query_jaccard", "lead_query_coverage"]
BIN_FEATURES = ["has_recent_year", "lead_has_year", "title_is_question",
                "title_has_number", "title_listicle"]
FEATURES = CONT_FEATURES + BIN_FEATURES


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    n = len(groups)
    if word.lower().endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _featurize(url, text, title, content_length, fetch_status):
    row = {"url_canonical": url, "content_length": content_length, "fetch_status": fetch_status}
    t = title or ""
    tw = t.split()
    row["title_len_words"] = len(tw)
    row["title_is_question"] = int(t.strip().endswith("?") or bool(
        re.match(r"^(how|what|why|when|which|who|can|does|is|are|should)\b", t.strip(), re.I)))
    row["title_has_number"] = int(bool(RE_NUMTOK.search(t)))
    row["title_listicle"] = int(bool(RE_LISTICLE.search(t)))
    row["title_tokens"] = " ".join(sorted({w.lower() for w in RE_WORD.findall(t)} - STOP))

    if not isinstance(text, str) or not text.strip():
        row["has_text"] = 0
        return row
    row["has_text"] = 1

    words_full = text.split()
    n_words = len(words_full)
    row["n_chars_text"] = len(text)
    row["n_words"] = n_words

    lines = text.split("\n")
    nonblank = [ln for ln in lines if ln.strip()]
    n_nb = max(1, len(nonblank))
    row["newlines_per_100w"] = 100.0 * len(nonblank) / max(1, n_words)
    row["short_line_share"] = sum(1 for ln in nonblank if len(ln.strip()) < 40) / n_nb
    row["bullet_line_share"] = sum(1 for ln in nonblank if RE_BULLET.match(ln)) / n_nb
    row["question_lines_per_100"] = 100.0 * sum(1 for ln in nonblank if ln.rstrip().endswith("?")) / n_nb
    first30 = nonblank[:30]
    row["nav_noise_share"] = sum(1 for ln in first30 if len(ln.split()) < 4) / max(1, len(first30))

    # lead = first LEAD_WORDS words taken from substantive lines (>=4 words), skipping nav crumbs
    lead_words = []
    for ln in nonblank:
        if len(ln.split()) >= 4:
            lead_words.extend(ln.split())
        if len(lead_words) >= LEAD_WORDS:
            break
    lead_words = lead_words[:LEAD_WORDS]
    lead = " ".join(lead_words)
    row["lead_num_token_share"] = sum(1 for w in lead_words if RE_NUMTOK.search(w)) / max(1, len(lead_words))
    row["lead_has_year"] = int(bool(RE_YEAR.search(lead)))
    row["lead_tokens"] = " ".join(sorted({w.lower() for w in RE_WORD.findall(lead)} - STOP))

    cap = text[:CAP_CHARS]
    cap_words = cap.split()
    n_cw = max(1, len(cap_words))
    row["digits_per_100w"] = 100.0 * sum(1 for w in cap_words if RE_NUMTOK.search(w)) / n_cw
    years = [int(y) for y in RE_YEAR.findall(cap)]
    row["dates_per_100w"] = 100.0 * (len(years) + len(RE_MONTH.findall(cap))) / n_cw
    row["max_year"] = max(years) if years else np.nan
    row["has_recent_year"] = int(bool(years) and max(years) >= 2024)

    sents = [s for s in RE_SENT_SPLIT.split(cap) if s.strip()]
    n_sents = max(1, len(sents))
    row["avg_sentence_len_w"] = n_cw / n_sents
    alpha = RE_WORD.findall(cap)
    n_alpha = max(1, len(alpha))
    row["avg_word_len"] = sum(len(w) for w in alpha) / n_alpha
    syl = sum(_syllables(w) for w in alpha[:4000])
    row["flesch"] = 206.835 - 1.015 * (n_cw / n_sents) - 84.6 * (syl / max(1, min(len(alpha), 4000)))

    caps_mid = tot_mid = 0
    for s in sents[:400]:
        ws = RE_WORD.findall(s)
        for w in ws[1:]:
            tot_mid += 1
            caps_mid += int(w[0].isupper())
    row["caps_mid_share"] = caps_mid / max(1, tot_mid)

    first1000 = [w.lower() for w in alpha[:1000]]
    row["ttr_1000"] = len(set(first1000)) / max(1, len(first1000))
    return row


def step1_url_features(force=False):
    out = os.path.join(OUT_DIR, "url_features.parquet")
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists")
        return
    pf = pq.ParquetFile(os.path.join(SILVER, "sources_scaleup.parquet"))
    cols = ["url_canonical", "cleaned_text", "title_from_html", "content_length", "fetch_status"]
    rows, done, t0 = [], 0, time.time()
    for batch in pf.iter_batches(batch_size=200, columns=cols):
        d = batch.to_pydict()
        for i in range(len(d["url_canonical"])):
            rows.append(_featurize(d["url_canonical"][i], d["cleaned_text"][i],
                                   d["title_from_html"][i], d["content_length"][i],
                                   d["fetch_status"][i]))
        done += len(d["url_canonical"])
        if done % 2000 < 200:
            print(f"  {done} urls, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"[ok] wrote {out} {df.shape}")


def step2_citation_events(force=False):
    out_ev = os.path.join(OUT_DIR, "citation_events.parquet")
    out_agg = os.path.join(OUT_DIR, "url_engine_agg.parquet")
    if os.path.exists(out_ev) and os.path.exists(out_agg) and not force:
        print(f"[skip] {out_ev} exists")
        return
    f1 = pd.read_parquet(os.path.join(GOLD, "flat_table1_responses.parquet"),
                         columns=["response_id", "query_id", "engine", "run_index", "n_sources_cited"])
    f2 = pd.read_parquet(os.path.join(GOLD, "flat_table2_sentences.parquet"),
                         columns=["response_id", "sentence_idx", "is_artifact"])
    f3 = pd.read_parquet(os.path.join(GOLD, "flat_table3_citations.parquet"))
    f4 = pd.read_parquet(os.path.join(GOLD, "flat_table4_entailment_scores.parquet"))

    f4 = f4.merge(f2, on=["response_id", "sentence_idx"], how="left", validate="m:1")
    assert f4.is_artifact.notna().all(), "unmatched sentence in flat_table4"
    pairs = f4[~f4.is_artifact]
    print(f"  pairs: total={len(f4):,} non-artifact={len(pairs):,}")

    pe = (pairs.assign(sup=(pairs.entail_score >= TAU_SUP).astype(float),
                       con=(pairs.entail_score < TAU_CON).astype(float))
          .groupby(["response_id", "citation_position"], sort=False)
          .agg(pairs_n=("entail_score", "size"), mean_entail=("entail_score", "mean"),
               supported_share=("sup", "mean"), contradiction_share=("con", "mean"))
          .reset_index())

    ev = f3.merge(f1, on="response_id", how="left", validate="m:1")
    assert ev.engine.notna().all(), "citation with unknown response"
    ev = ev.merge(pe, on=["response_id", "citation_position"], how="left")
    ev["n_cited_in_response"] = ev.groupby("response_id")["citation_position"].transform("size")
    ev["is_first"] = (ev.citation_position == 1).astype(int)
    ev["is_early3"] = (ev.citation_position <= 3).astype(int)
    den = (ev.n_cited_in_response - 1).clip(lower=1)
    ev["rel_position"] = (ev.citation_position - 1) / den
    ev["fetch_ok"] = (ev.fetch_status == "ok").astype(int)
    ev.to_parquet(out_ev, index=False)
    print(f"[ok] wrote {out_ev} {ev.shape} | entail coverage {ev.pairs_n.notna().mean():.3f}")

    agg = (ev.groupby(["url_canonical", "engine"], sort=False)
           .agg(n_events=("response_id", "size"), n_queries=("query_id", "nunique"),
                mean_position=("citation_position", "mean"),
                mean_rel_position=("rel_position", "mean"),
                first_share=("is_first", "mean"), early3_share=("is_early3", "mean"),
                mean_supported_share=("supported_share", "mean"),
                mean_entail=("mean_entail", "mean"), domain=("domain", "first"))
           .reset_index())
    agg.to_parquet(out_agg, index=False)
    print(f"[ok] wrote {out_agg} {agg.shape}")


def step3_model_frames(force=False):
    out_m = os.path.join(OUT_DIR, "model_frame_events.parquet")
    if os.path.exists(out_m) and not force:
        print(f"[skip] {out_m} exists")
        return
    ev = pd.read_parquet(os.path.join(OUT_DIR, "citation_events.parquet"))
    uf = pd.read_parquet(os.path.join(OUT_DIR, "url_features.parquet"))
    enr = pd.read_parquet(os.path.join(ENRICH, "citations_scaleup_enriched.parquet"),
                          columns=["query_id", "engine", "run_index", "position", "domain_type"])
    tax = pd.read_parquet(os.path.join(ENRICH, "domain_taxonomy.parquet"))
    qf = pd.read_parquet(os.path.join(QUERIES, "reclass_queries.parquet"))

    ev = ev.merge(enr.rename(columns={"position": "citation_position"}),
                  on=["query_id", "engine", "run_index", "citation_position"],
                  how="left", validate="1:1")
    bad = ev[ev.domain_type.isna()]
    assert bad.domain.isna().all() and len(bad) <= 2, f"enrichment join failed: {len(bad)}"
    ev["domain_type"] = ev.domain_type.fillna("other")
    print(f"  [warn] {len(bad)} degenerate citation(s) with NaN domain -> 'other' (expected)")
    ev = ev.merge(tax[["domain", "pay_level_domain"]].drop_duplicates("domain"),
                  on="domain", how="left")
    ev["pay_level_domain"] = ev.pay_level_domain.fillna(ev.domain)
    ev = ev.merge(qf[["query_id", "query_text", "intent", "domain", "temporality"]]
                  .rename(columns={"domain": "query_domain"}),
                  on="query_id", how="left", validate="m:1")
    ev = ev.merge(uf, on="url_canonical", how="left", validate="m:1", suffixes=("", "_uf"))

    qtok = {r.query_id: set(w.lower() for w in RE_WORD.findall(r.query_text)) - STOP
            for r in qf.itertuples()}

    def jacc(a, b):
        return len(a & b) / len(a | b) if a and b else 0.0

    ttok = ev.title_tokens.fillna("").str.split().map(set)
    ltok = ev.lead_tokens.fillna("").str.split().map(set)
    qs = ev.query_id.map(qtok)
    ev["title_query_jaccard"] = [jacc(q, t) for q, t in zip(qs, ttok)]
    ev["lead_query_coverage"] = [len(q & l) / max(1, len(q)) if isinstance(q, set) else 0.0
                                 for q, l in zip(qs, ltok)]
    ev["log1p_n_words"] = np.log1p(ev.n_words)
    ev["flesch_c"] = ev.flesch.clip(-100, 121)

    m = ev[ev.has_text == 1].copy()
    print(f"  modeling events: {len(m):,} of {len(ev):,}")
    for c in CONT_FEATURES:  # winsorize 1/99 pooled, then z-standardize pooled
        lo, hi = m[c].quantile([0.01, 0.99])
        m[c] = m[c].clip(lo, hi)
        sd = m[c].std()
        m[c] = (m[c] - m[c].mean()) / (sd if sd > 0 else 1.0)
    for c in BIN_FEATURES:
        m[c] = m[c].fillna(0).astype(float)

    well = qf.domain.value_counts()
    well = set(well[well >= 8].index)
    m["qd"] = np.where(m.query_domain.isin(well), m.query_domain, "other_topic")
    top_types = ev.domain_type.value_counts().head(15).index
    m["dt"] = np.where(m.domain_type.isin(top_types), m.domain_type, "other_type")
    m["log_n_cited"] = np.log(m.n_cited_in_response)
    m.drop(columns=["title_tokens", "lead_tokens"]).to_parquet(out_m, index=False)
    print(f"[ok] wrote {out_m} {m.shape}")

    st = (m[m.engine != "gemini"]  # gemini page identity unobservable (redirect urls)
          .groupby(["url_canonical", "engine", "query_id"], sort=False)
          .agg(runs_cited=("run_index", "nunique"),
               pay_level_domain=("pay_level_domain", "first"),
               qd=("qd", "first"), dt=("dt", "first"), intent=("intent", "first"),
               log_n_cited=("log_n_cited", "mean"),
               **{f: (f, "first") for f in FEATURES})
          .reset_index())
    st["stickiness"] = st.runs_cited / 8.0
    st.to_parquet(os.path.join(OUT_DIR, "model_frame_sticky.parquet"), index=False)
    print(f"[ok] page stickiness rows (no gemini): {len(st):,}; mean {st.stickiness.mean():.3f}")

    std = (m.groupby(["pay_level_domain", "engine", "query_id"], sort=False)
           .agg(runs_cited=("run_index", "nunique"),
                qd=("qd", "first"), dt=("dt", "first"), intent=("intent", "first"),
                log_n_cited=("log_n_cited", "mean"),
                **{f: (f, "mean") for f in FEATURES})
           .reset_index())
    std["stickiness"] = std.runs_cited / 8.0
    std.to_parquet(os.path.join(OUT_DIR, "model_frame_sticky_domain.parquet"), index=False)
    print(f"[ok] domain stickiness rows (all engines): {len(std):,}; mean {std.stickiness.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    step1_url_features(args.force)
    step2_citation_events(args.force)
    step3_model_frames(args.force)
    print("done.")
