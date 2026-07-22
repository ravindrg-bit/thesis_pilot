"""
run_collection.py — engine-agnostic k-repeat collection loop (V3 Stage 5).

Loops the frozen query registry x engines x K runs, calling each adapter's fetch() (which
writes one bronze JSON per cell). Resumable, idempotent, profile-aware. COLLECTION ONLY —
normalise()/citation extraction happens later at silver, not here.

  - RESUME BY BRONZE FILE: a cell whose {engine}__{query_id}__r{run}.json already exists is
    SKIPPED (never re-called, never re-paid). Re-running resumes exactly where it stopped and
    naturally retries previously-failed cells (their bronze was never written).
  - QUERY-MAJOR (query -> engine -> run): keeps a query's engines close in time (cross-engine
    comparability on the same web state) AND a cell's K runs back-to-back (CV reflects model
    stochasticity, not web drift).
  - RETRY WITH BACKOFF on any exception; after MAX_RETRIES the cell is logged failed and the
    run continues.
  - PROFILE-AWARE: prints the profile banner; refuses a 'scaleup' run without --confirm-scaleup.
"""

import argparse
import json
import random
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env import load_env
load_env()

import pandas as pd

import thesis_config as cfg
from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.claude_adapter import ClaudeAdapter
from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.perplexity_adapter import PerplexityAdapter
from src.adapters.kimi_adapter import KimiAdapter

# MistralAdapter intentionally not imported: mistral is excluded from collection.
# Rationale: 218×HTTP 429 in pilot → 82/300 captured, not comparable.
# See docs/decision_logs/2026-06-06_mistral_excluded.md
# src/adapters/mistral_adapter.py is RETAINED for pilot reproducibility.

ADAPTERS = {
    "perplexity": PerplexityAdapter,
    "claude": ClaudeAdapter,
    "chatgpt": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "kimi": KimiAdapter,
}

MAX_RETRIES = 4
BACKOFF_BASE = 2.0
BACKOFF_CAP = 30.0
INTER_CALL_DELAY = 0.5

# Per-engine pacing overrides (any engine not listed uses the defaults above).
# Mistral: the Agents API enforces a hard 1 request/second cap, and the web_search connector
# fans a single cell into several rapid back-end calls -> bursts trip HTTP 429 (the SDK
# surfaces these as a generic SDKError). TPM (50k) has ample headroom (median ~520 tok/cell),
# so only the request rate is throttled. The inter-call delay is ADAPTIVE (AIMD): it starts at
# inter_call_delay and, on a failed cell, multiplicatively backs off toward max_delay (slower);
# after a clean window of successes it eases back down toward the floor (faster). So the speed
# tracks the live failure rate instead of being fixed. backoff_* applies to the per-cell retry.
DEFAULT_PACING = {"inter_call_delay": INTER_CALL_DELAY, "max_delay": INTER_CALL_DELAY,
                  "backoff_base": BACKOFF_BASE, "backoff_cap": BACKOFF_CAP,
                  "backoff_floor": 0.0, "jitter": 0.0}
ENGINE_PACING = {
    "mistral": {"inter_call_delay": 2.0, "max_delay": 20.0, "backoff_base": 4.0,
                "backoff_cap": 30.0, "backoff_floor": 15.0, "jitter": 5.0},
}

# Adaptive inter-call-delay controller (AIMD): on each failed cell multiply the delay by
# ADAPT_UP (capped at the engine's max_delay); after a full clean window of ADAPT_WINDOW
# consecutive successes multiply by ADAPT_DOWN (floored at inter_call_delay). max_delay ==
# inter_call_delay (the default) disables adaptation, so only mistral adapts.
ADAPT_WINDOW = 8
ADAPT_UP = 1.5
ADAPT_DOWN = 0.85

# Circuit-breaker: if one engine returns this many consecutive failures of the SAME type
# (e.g. 20 straight HTTP 429s), its run is aborted rather than grinding through every
# remaining cell against an exhausted quota. Per-engine and resume-safe: other engines are
# unaffected, no bronze is touched, and rerunning resumes from bronze once the window resets.
CIRCUIT_BREAKER_N = 20


def _bronze_path(engine, qid, run):
    return cfg.BRONZE / f"{engine}__{qid}__r{run}.json"


def _approx_tokens(usage):
    if not isinstance(usage, dict):
        return 0
    for k in ("total_tokens", "totalTokenCount", "total_token_count"):
        if isinstance(usage.get(k), int):
            return usage[k]
    return sum(v for k, v in usage.items() if k.endswith("tokens") and isinstance(v, int))


def _http_status(exc):
    """Best-effort HTTP status code from an SDK exception (for the failure log / 429 confirmation)."""
    for attr in ("status_code", "status", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "raw_response", None) or getattr(exc, "response", None)
    if resp is not None and isinstance(getattr(resp, "status_code", None), int):
        return resp.status_code
    return None


def _call_with_retry(adapter, qid, qtext, run, pacing):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return adapter.fetch(qid, qtext, run), None
        except Exception as e:
            last = e
            if attempt < MAX_RETRIES:
                wait = min(pacing["backoff_base"] ** attempt, pacing["backoff_cap"])
                wait = max(wait, pacing["backoff_floor"]) + random.uniform(0, pacing["jitter"])
                code = _http_status(e)
                label = type(e).__name__ + (f" HTTP {code}" if code else "")
                print(f"      retry after {label} (waiting {wait:.0f}s)")
                time.sleep(wait)
    return None, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="", help="comma-list subset; default = all in cfg.ENGINES")
    ap.add_argument("--engine", default=None,
                    help="restrict the run to a single engine key (chatgpt|claude|gemini|perplexity|kimi). Default: all five.")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N queries (smoke test)")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N queries (use with --limit to split across pods)")
    ap.add_argument("--confirm-scaleup", action="store_true", help="required for the scaleup profile")
    args = ap.parse_args()

    print("=" * 72)
    print(cfg.profile_summary())
    print("=" * 72)

    if cfg.RUN_PROFILE == "scaleup" and not args.confirm_scaleup:
        print("REFUSING: scaleup profile requires --confirm-scaleup (the large, costly run).")
        print("Re-run:  THESIS_RUN_PROFILE=scaleup python scripts/run_collection.py --confirm-scaleup")
        sys.exit(2)

    if not cfg.QUERY_REGISTRY.exists():
        print(f"[FAIL] frozen registry not found at {cfg.QUERY_REGISTRY}.")
        sys.exit(1)

    df = pd.read_parquet(cfg.QUERY_REGISTRY)
    if args.offset and args.offset > 0:
        df = df.iloc[args.offset :]
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit]

    if args.engine is not None and args.engine not in cfg.ENGINES:
        print(f"[FAIL] unknown engine '{args.engine}'. Valid keys: {list(cfg.ENGINES.keys())}")
        sys.exit(1)
    if args.engine:
        engines = [args.engine]
    else:
        engines = [e.strip() for e in args.engines.split(",") if e.strip()] or list(cfg.ENGINES.keys())
    for e in engines:
        if e not in ADAPTERS:
            print(f"[FAIL] unknown engine '{e}'. Known: {list(ADAPTERS.keys())}")
            sys.exit(1)

    cfg.BRONZE.mkdir(parents=True, exist_ok=True)
    K, n = cfg.K, len(df)
    total_cells = n * len(engines) * K
    print(f"queries={n}  engines={engines}  K={K}  ->  {total_cells} cells")
    print(f"bronze dir: {cfg.BRONZE}")
    for e in engines:
        p = ENGINE_PACING.get(e)
        if p:
            print(f"[pacing] {e}: inter_call_delay={p['inter_call_delay']}s (adaptive, max {p['max_delay']}s)  "
                  f"backoff floor={p['backoff_floor']}s cap={p['backoff_cap']}s +jitter≤{p['jitter']}s  "
                  f"(rate-limit throttle)")
    print("-" * 72)

    instances, init_errors = {}, {}
    for e in engines:
        try:
            instances[e] = ADAPTERS[e]()
        except Exception as ex:
            init_errors[e] = f"{type(ex).__name__}: {ex}"
            print(f"[engine init FAILED] {e}: {init_errors[e]} -> its cells will be marked failed")

    tally = {e: {"done": 0, "skipped": 0, "failed": 0, "tokens": 0} for e in engines}
    # per-engine adaptive inter-call-delay state (AIMD); see ADAPT_* and ENGINE_PACING
    adapt = {e: {"delay": ENGINE_PACING.get(e, DEFAULT_PACING)["inter_call_delay"],
                 "recent": deque(maxlen=ADAPT_WINDOW)} for e in engines}
    # per-engine circuit-breaker state: consecutive same-type failures; see CIRCUIT_BREAKER_N
    consec = {e: {"sig": None, "count": 0, "desc": ""} for e in engines}
    aborted = set()  # engines stopped early by the circuit-breaker
    failures = []
    started = datetime.now(timezone.utc)

    start = time.time()
    attempted = 0  # cells attempted this invocation (done + skipped + failed)

    def _progress(done_n, total, engine, qi, skipped, failed, tokens):
        elapsed = time.time() - start
        rate = done_n / elapsed if elapsed > 0 else 0          # cells/sec
        remaining = (total - done_n) / rate if rate > 0 else 0
        pct = 100.0 * done_n / total if total else 0
        bar_len = 24
        filled = int(bar_len * done_n / total) if total else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = (f"\r[{bar}] {done_n}/{total} ({pct:4.1f}%) "
               f"| now: {engine}:{qi} "
               f"| skip {skipped} fail {failed} "
               f"| elapsed {elapsed/60:5.1f}m eta {remaining/60:5.1f}m "
               f"| ~{tokens} tok ")
        sys.stderr.write(msg)
        sys.stderr.flush()

    for qi in range(n):
        row = df.iloc[qi]
        qid, qtext = row["query_id"], row["query_text"]
        for e in engines:
            if e in aborted:
                continue
            pacing = ENGINE_PACING.get(e, DEFAULT_PACING)
            for run in range(1, K + 1):
                tag = f"[q {qi+1}/{n} {qid}] {e} r{run}"
                path = _bronze_path(e, qid, run)
                if path.exists():
                    tally[e]["skipped"] += 1
                    print(f"{tag}  skip (exists)")
                elif e in init_errors:
                    tally[e]["failed"] += 1
                    failures.append({"engine": e, "query_id": qid, "run": run, "error": init_errors[e]})
                    print(f"{tag}  FAIL (no adapter)")
                else:
                    rec, err = _call_with_retry(instances[e], qid, qtext, run, pacing)
                    ok = rec is not None
                    if not ok:
                        tally[e]["failed"] += 1
                        code = _http_status(err)
                        failures.append({"engine": e, "query_id": qid, "run": run,
                                         "error": f"{type(err).__name__}: {err}", "http_status": code})
                        print(f"{tag}  FAIL after {MAX_RETRIES} attempts: {type(err).__name__}"
                              + (f" HTTP {code}" if code else ""))
                        sig = (type(err).__name__, code)
                        cb = consec[e]
                        cb["count"] = cb["count"] + 1 if sig == cb["sig"] else 1
                        cb["sig"] = sig
                        cb["desc"] = f"{type(err).__name__}" + (f" HTTP {code}" if code else "")
                    else:
                        tally[e]["done"] += 1
                        tok = _approx_tokens(rec.usage)
                        tally[e]["tokens"] += tok
                        print(f"{tag}  ok  ({len(rec.answer_text or '')} chars, ~{tok} tok)")
                        consec[e] = {"sig": None, "count": 0, "desc": ""}
                    # adaptive throttle: slow down on a failure, ease back up after a clean window
                    st = adapt[e]
                    st["recent"].append(0 if ok else 1)
                    prev = st["delay"]
                    if not ok:
                        st["delay"] = min(st["delay"] * ADAPT_UP, pacing["max_delay"])
                    elif len(st["recent"]) == ADAPT_WINDOW and sum(st["recent"]) == 0:
                        st["delay"] = max(st["delay"] * ADAPT_DOWN, pacing["inter_call_delay"])
                    if abs(st["delay"] - prev) > 1e-9:
                        print(f"      [adapt] {e} inter_call_delay {prev:.1f}s -> {st['delay']:.1f}s "
                              f"(recent fails {sum(st['recent'])}/{len(st['recent'])})")
                    time.sleep(st["delay"])
                attempted += 1
                _progress(attempted, total_cells, e, qi + 1,
                          sum(t["skipped"] for t in tally.values()),
                          sum(t["failed"] for t in tally.values()),
                          sum(t["tokens"] for t in tally.values()))
                if consec[e]["count"] >= CIRCUIT_BREAKER_N:
                    aborted.add(e)
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    print(f"[CIRCUIT-BREAKER] {e}: {consec[e]['count']} consecutive "
                          f"{consec[e]['desc']} errors -- aborting {e}'s run; rerun to resume "
                          f"from bronze after the window resets.")
                    break

    sys.stderr.write("\n")
    sys.stderr.flush()

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    if failures:
        log = cfg.DATA / "collection_failures.jsonl"
        with open(log, "w") as fh:
            for f in failures:
                fh.write(json.dumps(f) + "\n")

    print("-" * 72)
    print(f"SUMMARY  profile={cfg.RUN_PROFILE}  elapsed={elapsed/60:.1f} min")
    print(f"{'engine':<12}{'done':>7}{'skip':>7}{'fail':>7}{'~tokens':>12}")
    tot = {"done": 0, "skipped": 0, "failed": 0, "tokens": 0}
    for e in engines:
        t = tally[e]
        print(f"{e:<12}{t['done']:>7}{t['skipped']:>7}{t['failed']:>7}{t['tokens']:>12}")
        for k in tot:
            tot[k] += t[k]
    print(f"{'TOTAL':<12}{tot['done']:>7}{tot['skipped']:>7}{tot['failed']:>7}{tot['tokens']:>12}")
    print(f"cells expected={total_cells}  accounted={tot['done']+tot['skipped']+tot['failed']}")
    if aborted:
        for e in sorted(aborted):
            print(f"[CIRCUIT-BREAKER] {e} aborted early after {consec[e]['count']} consecutive "
                  f"{consec[e]['desc']} errors -- remaining cells were NOT attempted.")
    if failures:
        print(f"\n{len(failures)} failures logged to {cfg.DATA / 'collection_failures.jsonl'}")
        print("Re-run the SAME command to retry only the failed/missing cells (resume-by-bronze-file).")
    else:
        print("\nAll cells accounted for, no failures.")


if __name__ == "__main__":
    main()
