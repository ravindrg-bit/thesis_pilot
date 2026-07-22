"""
nli_attribution.py - Option 3 attribution: NLI classifier replaces the LLM judge.

For each (sentence, source) pair: chunk the source into overlapping windows, run
NLI(premise=window, hypothesis=sentence) for each window, take the MAX entailment
probability across windows ("any passage of the source entails the sentence" - the
canonical aggregation, per Gao et al. 2023 / ALCE). If max-entailment >= threshold,
the source is judged to support the sentence.

Outputs the SAME shape as src.attribution.attribute_cell so PAWC and AIS values are
directly comparable across methods. PAWC / AIS arithmetic is unchanged - only the
support-decision step swaps.

Lineage: Rashkin et al. 2023 (AIS), Gao et al. 2023 (ALCE), Honovich et al. 2022 (TRUE).
"""

import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import thesis_config as cfg
from src.attribution import (
    segment_sentences, fetch_source_text, position_weight,
)

# ----- lazy-loaded NLI model ------------------------------------------------
_NLI = {"tok": None, "model": None, "entail_idx": None}


def _load_nli(model_name=None):
    if _NLI["model"] is not None:
        return
    name = model_name or cfg.NLI_MODEL
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(cfg.NLI_DEVICE)
    model.eval()
    # locate the 'entailment' label index from id2label (varies by checkpoint)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    entail_idx = next((i for i, v in id2label.items() if "entail" in v), 0)
    _NLI["tok"], _NLI["model"], _NLI["entail_idx"] = tok, model, entail_idx


def _chunk_source(text, tok, chunk_tokens=None, stride=None):
    """Split source text into overlapping token windows. Returns list of decoded strings."""
    ct = chunk_tokens or cfg.NLI_SOURCE_CHUNK_TOKENS
    st = stride or cfg.NLI_SOURCE_CHUNK_STRIDE
    ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(ids) <= ct:
        return [text]
    chunks = []
    i = 0
    while i < len(ids):
        chunk_ids = ids[i:i + ct]
        chunks.append(tok.decode(chunk_ids, skip_special_tokens=True))
        if i + ct >= len(ids):
            break
        i += (ct - st)
    return chunks


def _max_entailment(sentence, source_text, batch_size=8):
    """NLI(premise=source_chunk, hypothesis=sentence) over all chunks; return max P(entail)."""
    tok, model, eidx = _NLI["tok"], _NLI["model"], _NLI["entail_idx"]
    chunks = _chunk_source(source_text, tok)
    best = 0.0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        enc = tok(batch, [sentence] * len(batch), return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(cfg.NLI_DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[:, eidx].cpu().tolist()
        m = max(probs) if probs else 0.0
        if m > best:
            best = m
    return best


def attribute_cell_nli(answer_text, cited_sources, cache, threshold=None):
    """Option-3 attribution. Same return shape as src.attribution.attribute_cell.

    Decision rule per (sentence, source): max-entailment over chunked source >= threshold
    -> the source SUPPORTS the sentence. PAWC and AIS computed identically downstream.
    """
    _load_nli()
    thresh = threshold if threshold is not None else cfg.NLI_ENTAILMENT_THRESHOLD

    sentences = segment_sentences(answer_text)
    n = len(sentences)

    fetched, fetch_status = [], []
    for s in cited_sources:
        txt, status = fetch_source_text(s["url"], cache)
        fetch_status.append({"position": s["position"], "status": status})
        if txt:
            fetched.append((s["position"], txt))

    pawc = {pos: 0.0 for pos, _ in fetched}
    supported_sentences = 0
    nli_calls = 0

    sentence_detail = []

    for i, sent in enumerate(sentences, start=1):
        w = position_weight(i, n)
        wc = len(sent.split())
        supporting_positions = []
        source_scores = []
        for pos, src_text in fetched:
            score = _max_entailment(sent, src_text)
            nli_calls += 1
            source_scores.append({"pos": pos, "score": round(score, 5)})
            if score >= thresh:
                supporting_positions.append(pos)
        if supporting_positions:
            supported_sentences += 1
        for pos in supporting_positions:
            pawc[pos] += wc * w
        sentence_detail.append({
            "text": sent,
            "idx": i,
            "wc": wc,
            "w": round(w, 4),
            "supported": bool(supporting_positions),
            "sources": supporting_positions,
            "source_scores": source_scores
        })

    return {
        "n_sentences": n,
        "n_sources_cited": len(cited_sources),
        "n_sources_fetched_ok": len(fetched),
        "fetch_status": fetch_status,
        "ais_supported_sentences": supported_sentences,
        "ais_rate": (supported_sentences / n) if n else None,
        "sentence_detail": sentence_detail,
        "pawc_by_source_position": {str(k): round(v, 3) for k, v in pawc.items()},
        "pawc_total": round(sum(pawc.values()), 3),
        "nli_evaluations": nli_calls,      # analogue of judge_calls; no $ cost
        "input_tokens": 0,                 # not API tokens; included for schema parity
        "output_tokens": 0,
    }
