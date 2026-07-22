"""
build_silver.py — CLI entry point for the main-pipeline SILVER build.

Thin wrapper around src.silver.build_silver(): reads every bronze JSON under the
ACTIVE run profile's tree (cfg.BRONZE), normalises each record through its engine
adapter, and writes the two analysis tables —

  data/<profile>/silver/responses_<profile>.parquet   (one row per engine×query×run)
  data/<profile>/silver/citations_<profile>.parquet   (one row per cited source)

The run profile is chosen per-invocation by the THESIS_RUN_PROFILE env var and is
NEVER defaulted here: the script refuses to run unless it is set explicitly, so a
scaleup build can never happen by accident (and a pilot build is equally deliberate).
No API keys or network needed — normalise() reads only the captured bronze.

Run:
  THESIS_RUN_PROFILE=pilot   python scripts/build_silver.py
  THESIS_RUN_PROFILE=scaleup python scripts/build_silver.py
"""

import os
import sys
from pathlib import Path

# --- project root on sys.path (house pattern) -------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- require an EXPLICIT profile BEFORE anything reads the config ------------
# thesis_config silently defaults THESIS_RUN_PROFILE to "pilot"; we refuse that
# implicit default here so the target data tree is always a conscious choice.
RUN_PROFILE = os.environ.get("THESIS_RUN_PROFILE")
if not RUN_PROFILE:
    sys.stderr.write(
        "[STOP] THESIS_RUN_PROFILE is not set — refusing to build silver under an\n"
        "       implicit profile (would silently target the pilot tree).\n"
        "       Set it explicitly, e.g.:\n"
        "         THESIS_RUN_PROFILE=pilot   python scripts/build_silver.py\n"
        "         THESIS_RUN_PROFILE=scaleup python scripts/build_silver.py\n"
    )
    sys.exit(1)

from src.env import load_env
load_env()

import thesis_config as cfg
from src.silver import build_silver


def main() -> None:
    resp_out = cfg.SILVER / f"responses_{cfg.RUN_PROFILE}.parquet"
    cit_out = cfg.SILVER / f"citations_{cfg.RUN_PROFILE}.parquet"

    print("=" * 72)
    print("SILVER BUILD")
    print(cfg.profile_summary())
    print(f"  bronze in : {cfg.BRONZE}")
    print(f"  silver out: {cfg.SILVER}")
    print(f"  -> {resp_out.name}")
    print(f"  -> {cit_out.name}")
    print("=" * 72)

    responses, citations, report = build_silver(write=True)

    print("BUILD REPORT")
    for k, v in report.items():
        if k != "failures":
            print(f"  {k}: {v}")
    print(f"  failures: {len(report['failures'])}")
    for f in report["failures"][:10]:
        print("   ", f)

    print("-" * 72)
    print(f"written: {resp_out.relative_to(cfg.ROOT)}  ({len(responses)} rows)")
    print(f"         {cit_out.relative_to(cfg.ROOT)}  ({len(citations)} rows)")
    print("=" * 72)


if __name__ == "__main__":
    main()
