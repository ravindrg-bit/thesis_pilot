"""
attribution.py - shared support-attribution backbone for PAWC (RQ1) and AIS (RQ4).

For each answer sentence, ONE judge call (Claude Haiku 4.5) decides which of the answer's
cited sources support it - by READING fetched source snippets, NOT native citation markers.
Uniform across all engines (incl. perplexity/kimi, which emit no spans).

  AIS  = supported_sentences / total_sentences   (supported = any cited source supports it)
  PAWC(source) = sum over supported sentences of (word_count x position_weight)
                 [every supporting source gets full credit; linear-decay position weight]

Lineage: Aggarwal 2024, Rashkin 2023, Gao 2023, Luttgenau 2025. Support-based = documented
deviation from Aggarwal's declared attribution. COST: fetches pages + paid judge (not offline).
This module's judge call is SYNCHRONOUS (smoke test / pilot); batch is a later swap.
"""

import re

import requests
from bs4 import BeautifulSoup

import thesis_config as cfg
from src.silver import canonicalise_url

_UA = {"User-Agent": "Mozilla/5.0 (thesis-research; PAWC/AIS attribution)"}
_STRIP_TAGS = ["script", "style", "nav", "header", "footer"]


def html_to_text(html: str) -> str:
    """Extract readable text from HTML: decompose noise tags, collapse whitespace.
    Shared with build_sources.py — keep cleaning rules in sync via this function."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def segment_sentences(text):
    """Split answer text into sentences using the PINNED backend (cfg.SENTENCE_SEGMENTER).

    The backend is part of the measurement instrument: N drives the AIS denominator and
    every PAWC position weight. Pilot gold was produced by the regex splitter (verified
    against stored n_sentences, 1500/1500 cells), so "regex" is the default for all
    profiles. Selection is EXPLICIT — there is deliberately no try/except fallback: if
    "punkt" is requested but the NLTK data is absent, this fails loudly rather than
    silently swapping the instrument mid-experiment."""
    text = (text or "").strip()
    if not text:
        return []
    if cfg.SENTENCE_SEGMENTER == "punkt":
        import nltk  # nltk.sent_tokenize raises LookupError if punkt data is missing
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def fetch_source_text(url, cache, char_limit=None):
    """Fetch + extract readable text, deduped via cache. Returns (text_or_None, status)."""
    char_limit = char_limit or cfg.JUDGE_SOURCE_CHAR_LIMIT
    key = canonicalise_url(url)
    if key in cache:
        return cache[key]
    try:
        r = requests.get(url, headers=_UA, timeout=20)
        if r.status_code != 200:
            result = (None, f"http_{r.status_code}")
        else:
            txt = html_to_text(r.text)
            result = ((txt[:char_limit] if txt else None), "ok" if txt else "empty")
    except Exception as e:
        result = (None, f"{type(e).__name__}")
    cache[key] = result
    return result


def position_weight(idx_1based, n_sentences):
    if n_sentences <= 0:
        return 0.0
    return (n_sentences - idx_1based + 1) / n_sentences


def judge_supporting_sources(sentence, source_texts, client, model=None):
    """ONE judge call. Returns (supporting_1based_indices, input_tokens, output_tokens)."""
    model = model or cfg.JUDGE_MODEL
    if not source_texts:
        return [], 0, 0
    blocks = [f"[SOURCE {i}]\n{txt}\n" for i, (_, txt) in enumerate(source_texts, start=1)]
    prompt = (
        "You verify citations. Decide which numbered sources DIRECTLY SUPPORT the statement "
        "(state or clearly imply its factual content). Topical relatedness is NOT support.\n\n"
        f"STATEMENT:\n{sentence}\n\n" + "\n".join(blocks) +
        "\nReturn ONLY a JSON array of supporting source numbers, e.g. [1,3], or [] if none."
    )
    try:
        msg = client.messages.create(
            model=model, max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        import json as _json
        nums = _json.loads(m.group(0)) if m else []
        idxs = [int(n) for n in nums if str(n).isdigit() or isinstance(n, int)]
        usage = getattr(msg, "usage", None)
        return idxs, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0)
    except Exception:
        return [], 0, 0


def attribute_cell(answer_text, cited_sources, client, cache, char_limit=None):
    """Core shared primitive for ONE cell. cited_sources: list of dicts {position, url}.
    Returns metrics + token tallies. PAWC keyed by source position."""
    sentences = segment_sentences(answer_text)
    n = len(sentences)

    fetched, fetch_status = [], []
    for s in cited_sources:
        txt, status = fetch_source_text(s["url"], cache, char_limit)
        fetch_status.append({"position": s["position"], "status": status})
        if txt:
            fetched.append((s["position"], txt))

    pawc = {pos: 0.0 for pos, _ in fetched}
    supported_sentences = 0
    in_tok = out_tok = n_calls = 0
    for i, sent in enumerate(sentences, start=1):
        w = position_weight(i, n)
        wc = len(sent.split())
        idxs, it, ot = judge_supporting_sources(sent, fetched, client)
        n_calls += 1
        in_tok += it
        out_tok += ot
        supporting_positions = [fetched[j - 1][0] for j in idxs if 1 <= j <= len(fetched)]
        if supporting_positions:
            supported_sentences += 1
        for pos in supporting_positions:
            pawc[pos] += wc * w

    return {
        "n_sentences": n,
        "n_sources_cited": len(cited_sources),
        "n_sources_fetched_ok": len(fetched),
        "fetch_status": fetch_status,
        "ais_supported_sentences": supported_sentences,
        "ais_rate": (supported_sentences / n) if n else None,
        "pawc_by_source_position": {str(k): round(v, 3) for k, v in pawc.items()},
        "pawc_total": round(sum(pawc.values()), 3),
        "judge_calls": n_calls,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def judge_sentences_batched(sentences, source_texts, client, model=None):
    """Judge ALL sentences of one answer in ONE call (sources sent once).
    Returns (mapping, input_tokens, output_tokens) where mapping is
    {sentence_index_1based: [supporting_source_1based_indices]}.
    Identical support semantics as judge_supporting_sources, just batched."""
    import json as _json, re
    model = model or cfg.JUDGE_MODEL
    if not source_texts or not sentences:
        return {}, 0, 0
    src_blocks = [f"[SOURCE {i}]\n{txt}\n" for i, (_, txt) in enumerate(source_texts, start=1)]
    sent_blocks = [f"{i}. {s}" for i, s in enumerate(sentences, start=1)]
    prompt = (
        "You verify citations. For EACH numbered statement, decide which numbered sources "
        "DIRECTLY SUPPORT it (state or clearly imply its factual content). Topical relatedness "
        "is NOT support.\n\n"
        "SOURCES:\n" + "\n".join(src_blocks) + "\n\nSTATEMENTS:\n" + "\n".join(sent_blocks) +
        '\n\nReturn ONLY a JSON object mapping each statement number (as a string) to an array '
        'of supporting source numbers, e.g. {"1":[1],"2":[1,2],"3":[]}. Include every statement.'
    )
    try:
        msg = client.messages.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        obj = _json.loads(m.group(0)) if m else {}
        mapping = {}
        for k, v in obj.items():
            if str(k).isdigit() and isinstance(v, list):
                mapping[int(k)] = [int(n) for n in v if str(n).isdigit() or isinstance(n, int)]
        usage = getattr(msg, "usage", None)
        return mapping, getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0)
    except Exception:
        return {}, 0, 0


def attribute_cell_batched(answer_text, cited_sources, client, cache, char_limit=None):
    """Same outputs/keys as attribute_cell, but ONE judge call for the whole answer."""
    sentences = segment_sentences(answer_text)
    n = len(sentences)
    fetched, fetch_status = [], []
    for s in cited_sources:
        txt, status = fetch_source_text(s["url"], cache, char_limit)
        fetch_status.append({"position": s["position"], "status": status})
        if txt:
            fetched.append((s["position"], txt))

    mapping, in_tok, out_tok = judge_sentences_batched(sentences, fetched, client)
    pawc = {pos: 0.0 for pos, _ in fetched}
    supported_sentences = 0
    for i, sent in enumerate(sentences, start=1):
        w = position_weight(i, n)
        wc = len(sent.split())
        idxs = mapping.get(i, [])
        supporting_positions = [fetched[j - 1][0] for j in idxs if 1 <= j <= len(fetched)]
        if supporting_positions:
            supported_sentences += 1
        for pos in supporting_positions:
            pawc[pos] += wc * w
    return {
        "n_sentences": n, "n_sources_cited": len(cited_sources),
        "n_sources_fetched_ok": len(fetched), "fetch_status": fetch_status,
        "ais_supported_sentences": supported_sentences,
        "ais_rate": (supported_sentences / n) if n else None,
        "pawc_by_source_position": {str(k): round(v, 3) for k, v in pawc.items()},
        "pawc_total": round(sum(pawc.values()), 3),
        "judge_calls": 1 if (fetched and n) else 0,
        "input_tokens": in_tok, "output_tokens": out_tok,
    }


def judge_sentences_chunked(sentences, source_texts, client, chunk_size=10, model=None):
    """Judge sentences in CHUNKS of chunk_size (sources re-sent once per chunk).
    Returns (mapping, input_tokens, output_tokens, n_calls) with the SAME support
    semantics as the per-sentence and full-batch versions; only batching granularity differs.
    chunk_size=1 reproduces per-sentence; chunk_size>=len(sentences) reproduces full-batch."""
    import json as _json, re
    model = model or cfg.JUDGE_MODEL
    if not source_texts or not sentences:
        return {}, 0, 0, 0
    src_blocks = [f"[SOURCE {i}]\n{txt}\n" for i, (_, txt) in enumerate(source_texts, start=1)]
    src_text = "SOURCES:\n" + "\n".join(src_blocks)
    mapping, in_tok, out_tok, n_calls = {}, 0, 0, 0
    for start in range(0, len(sentences), chunk_size):
        chunk = sentences[start:start + chunk_size]
        # number statements by their GLOBAL 1-based index so mapping keys are absolute
        sent_blocks = [f"{start + j + 1}. {s}" for j, s in enumerate(chunk)]
        prompt = (
            "You verify citations. For EACH numbered statement, decide which numbered sources "
            "DIRECTLY SUPPORT it (state or clearly imply its factual content). Topical "
            "relatedness is NOT support.\n\n"
            + src_text + "\n\nSTATEMENTS:\n" + "\n".join(sent_blocks) +
            '\n\nReturn ONLY a JSON object mapping each statement number (as a string) to an '
            'array of supporting source numbers, e.g. {"1":[1],"2":[1,2]}. Include every statement.'
        )
        try:
            msg = client.messages.create(
                model=model, max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            obj = _json.loads(m.group(0)) if m else {}
            for k, v in obj.items():
                if str(k).isdigit() and isinstance(v, list):
                    mapping[int(k)] = [int(n) for n in v if str(n).isdigit() or isinstance(n, int)]
            usage = getattr(msg, "usage", None)
            in_tok += getattr(usage, "input_tokens", 0)
            out_tok += getattr(usage, "output_tokens", 0)
            n_calls += 1
        except Exception:
            n_calls += 1
    return mapping, in_tok, out_tok, n_calls


def attribute_cell_chunked(answer_text, cited_sources, client, cache, chunk_size=10, char_limit=None):
    """Same output keys as attribute_cell / attribute_cell_batched, chunked judging."""
    sentences = segment_sentences(answer_text)
    n = len(sentences)
    fetched, fetch_status = [], []
    for s in cited_sources:
        txt, status = fetch_source_text(s["url"], cache, char_limit)
        fetch_status.append({"position": s["position"], "status": status})
        if txt:
            fetched.append((s["position"], txt))

    mapping, in_tok, out_tok, n_calls = judge_sentences_chunked(
        sentences, fetched, client, chunk_size=chunk_size)
    pawc = {pos: 0.0 for pos, _ in fetched}
    supported_sentences = 0
    for i, sent in enumerate(sentences, start=1):
        w = position_weight(i, n)
        wc = len(sent.split())
        idxs = mapping.get(i, [])
        supporting_positions = [fetched[j - 1][0] for j in idxs if 1 <= j <= len(fetched)]
        if supporting_positions:
            supported_sentences += 1
        for pos in supporting_positions:
            pawc[pos] += wc * w
    return {
        "n_sentences": n, "n_sources_cited": len(cited_sources),
        "n_sources_fetched_ok": len(fetched), "fetch_status": fetch_status,
        "ais_supported_sentences": supported_sentences,
        "ais_rate": (supported_sentences / n) if n else None,
        "pawc_by_source_position": {str(k): round(v, 3) for k, v in pawc.items()},
        "pawc_total": round(sum(pawc.values()), 3),
        "judge_calls": n_calls, "input_tokens": in_tok, "output_tokens": out_tok,
    }
