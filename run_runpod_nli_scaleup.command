#!/usr/bin/env bash
# RunPod NLI SCALEUP launcher — adapted from run_runpod_nli_pilot.command (kept for provenance).
# Runs from YOUR Mac (uses ~/.ssh/id_ed25519). Double-click in Finder, or:
#   bash "run_runpod_nli_scaleup.command"
#
# DIFFERENCES vs the pilot launcher (each one was a verified failure mode for scaleup):
#   1. Correct repo URL (origin is thesis_pilot.git, NOT thesis-data-pilot.git).
#   2. Pod code is force-synced to origin/main (clone-if-missing left stale code behind).
#   3. No sed of thesis_config.py — NLI_DEVICE / THESIS_RUN_PROFILE are env vars now.
#   4. rsyncs ONLY the three scaleup silver parquets (~3.5 GB). Bronze is NOT needed:
#      run_nli_pilot.py reads silver + sources parquet only.
#   5. Shutdown timer 48h, not 6h: scaleup is ~6.7x the pilot workload (9,992 vs 1,500
#      cells; pilot ETA was ~5h => expect ~30-36h on a comparable GPU).
#   6. run_nli_pilot.py now checkpoints every 200 cells to nli_scaleup.partial.parquet
#      and RESUMES from it — a shutdown costs <=200 cells, not the whole run.
#
# EDIT THESE THREE BEFORE RUNNING (from the RunPod Connect dialog):
POD_IP="EDIT_ME"
POD_PORT="EDIT_ME"
SSH_KEY="$HOME/.ssh/id_ed25519"

REPO="https://github.com/ravindrg-bit/thesis_pilot.git"
LOCAL_PROJECT="$HOME/Desktop/Thesis Data Pilot"
EXPECTED_GOLD_ROWS=9992      # 1,249 cells x 8 runs (kimi/gb_3fcf760b1a2ea4f8 missing — see decision log)

set -u
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$POD_PORT")
RSH="ssh -i $SSH_KEY -p $POD_PORT -o StrictHostKeyChecking=accept-new"

stop() { echo ""; echo "==== STOP: $* ===="; echo "Halting. Fix the above, then re-run."; exit 1; }
hr()   { echo "------------------------------------------------------------"; }

[ "$POD_IP" != "EDIT_ME" ] || stop "Edit POD_IP/POD_PORT at the top of this script first."

echo "RunPod NLI SCALEUP — starting at $(date)"
hr

# ================= STEP 1: SSH reachability =================
echo "STEP 1 — Verify SSH reachability"
nc -zv -w 5 "$POD_IP" "$POD_PORT" || stop "TCP to $POD_IP:$POD_PORT failed. Likely your network blocks high ports — switch to phone hotspot and re-run."

PROBE=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "echo CONNECTED && hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>&1)
echo "$PROBE"
echo "$PROBE" | grep -q "CONNECTED" || stop "SSH probe did not return CONNECTED. See output above."
hr

# ================= STEP 2: Pod environment setup =================
echo "STEP 2 — Set up pod environment (clone/sync to origin/main + venv + deps + model + cuda)"
ssh "${SSH_OPTS[@]}" root@"$POD_IP" bash -s <<REMOTE
set -e
cd /workspace
if [ ! -d thesis ]; then
  git clone $REPO thesis 2>&1
fi
cd /workspace/thesis
# ALWAYS sync to origin/main — a pre-existing clone must not run stale code.
git fetch origin && git reset --hard origin/main && git log --oneline -1
apt-get update -qq && apt-get install -y -qq tmux wget
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.lock --quiet
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT REACHABLE'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"
mkdir -p data/scaleup/silver data/scaleup/gold
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; m='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'; AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m); print('Model cached.')"
echo "STRK_COUNT: \$(grep -c 'str(k): round' src/nli_attribution.py src/attribution.py | awk -F: '{s+=\$2} END{print s}')"
REMOTE
RC=$?
[ $RC -eq 0 ] || stop "Pod setup failed (exit $RC). If you saw a Git auth prompt, the repo is private to the pod — make it public or use a PAT. If CUDA NOT REACHABLE, this pod is unusable."

# ---- Gates: config resolves to scaleup+cuda+regex via env vars; struct keys stringified ----
GATE=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "cd /workspace/thesis && source .venv/bin/activate && THESIS_RUN_PROFILE=scaleup NLI_DEVICE=cuda python -c \"
import thesis_config as cfg
assert cfg.RUN_PROFILE == 'scaleup', cfg.RUN_PROFILE
assert cfg.NLI_DEVICE == 'cuda', cfg.NLI_DEVICE
assert cfg.SENTENCE_SEGMENTER == 'regex', cfg.SENTENCE_SEGMENTER
print('CONFIG_GATE_OK', cfg.GOLD)\"")
echo "$GATE"
echo "$GATE" | grep -q "CONFIG_GATE_OK" || stop "Config gate failed: profile/device/segmenter did not resolve as expected."
STRK=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "grep -c 'str(k): round' /workspace/thesis/src/nli_attribution.py /workspace/thesis/src/attribution.py | awk -F: '{s+=\$2} END{print s}'")
[ "$STRK" = "4" ] || stop "Expected exactly 4 'str(k): round' lines on pod, found: $STRK"
echo "Gates OK: CUDA, profile=scaleup, device=cuda, segmenter=regex, str(k)=4"
hr

# ================= STEP 3: Upload scaleup silver (3 parquets, ~3.5 GB) =================
echo "STEP 3 — rsync scaleup silver parquets to pod (responses + citations + sources; NO bronze, NO sources_bronze/)"
rsync -avz --progress -e "$RSH" \
  "$LOCAL_PROJECT/data/scaleup/silver/responses_scaleup.parquet" \
  "$LOCAL_PROJECT/data/scaleup/silver/citations_scaleup.parquet" \
  "$LOCAL_PROJECT/data/scaleup/silver/sources_scaleup.parquet" \
  root@"$POD_IP":/workspace/thesis/data/scaleup/silver/ || stop "rsync silver failed. Retry once (rsync resumes)."

SLIST=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "ls -la /workspace/thesis/data/scaleup/silver")
echo "$SLIST"
for f in responses_scaleup.parquet citations_scaleup.parquet sources_scaleup.parquet; do
  echo "$SLIST" | grep -q "$f" || stop "Missing on pod: $f"
done
echo "Upload verified: all 3 silver parquets on pod"
hr

# ================= STEP 4: CRITICAL GATE — single-cell write test (scaleup profile) =================
echo "STEP 4 — single-cell NLI + parquet write test under THESIS_RUN_PROFILE=scaleup"
WRITE_OUT=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "cd /workspace/thesis && source .venv/bin/activate && THESIS_RUN_PROFILE=scaleup NLI_DEVICE=cuda python -" <<'PY' 2>&1
import sys
sys.path.insert(0, '.')
from src.env import load_env
load_env()
import pandas as pd
import thesis_config as cfg
from src import nli_attribution as nli
from src.silver import canonicalise_url

responses = pd.read_parquet(cfg.SILVER / f'responses_{cfg.RUN_PROFILE}.parquet')
citations = pd.read_parquet(cfg.SILVER / f'citations_{cfg.RUN_PROFILE}.parquet')
# prefetch the one cell's sources from the parquet — NO live fetch even in the test
cell = responses[(responses.engine == 'claude')].iloc[0]
qid, run = cell['query_id'], int(cell['run_index'])
answer = cell['answer_text']
srcs = (citations[(citations.engine == 'claude') & (citations.query_id == qid) & (citations.run_index == run)]
        .sort_values('position')[['position', 'url']].to_dict('records'))
# STREAM the sources parquet (cleaned_text uncompressed ~6.4 GB; a whole-file read
# OOM-kills a 16 GB machine) — keep only the URLs this one test cell needs.
import pyarrow.parquet as pq
need = {canonicalise_url(s['url']) for s in srcs}
cache = {}
pf = pq.ParquetFile(cfg.SILVER / f'sources_{cfg.RUN_PROFILE}.parquet')
for batch in pf.iter_batches(batch_size=256, columns=['url_canonical', 'fetch_status', 'cleaned_text']):
    for url_c, status, text in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist()):
        k = canonicalise_url(url_c)
        if k in need:
            cache[k] = (text[:cfg.JUDGE_SOURCE_CHAR_LIMIT], 'ok') if (status == 'ok' and text) else (None, status)
res = nli.attribute_cell_nli(answer, srcs, cache)
res.update({'engine': 'claude', 'query_id': qid, 'run_index': run})
df = pd.DataFrame([res])
test_path = cfg.GOLD / 'TEST_scaleup_write.parquet'
df.to_parquet(test_path, index=False)
df2 = pd.read_parquet(test_path)
keys = list(df2.iloc[0]['pawc_by_source_position'].keys())
print(f'WRITE_OK={len(df2)==1} | KEYS_ALL_STRINGS={all(isinstance(k, str) for k in keys)} | CACHE_HITS={len(cache)} | SEGMENTER={cfg.SENTENCE_SEGMENTER}')
PY
)
echo "$WRITE_OUT"
echo "$WRITE_OUT" | grep -q "WRITE_OK=True | KEYS_ALL_STRINGS=True" || stop "Write test did not pass. DO NOT launch. Traceback is above."
ssh "${SSH_OPTS[@]}" root@"$POD_IP" "rm -f /workspace/thesis/data/scaleup/gold/TEST_scaleup_write.parquet"
echo "Write test PASSED and test artifact cleaned."
hr

# ================= STEP 5: Launch scaleup + 48h hard-shutdown insurance =================
echo "STEP 5 — Launch scaleup NLI in background with 48h hard-shutdown timer"
LAUNCH_OUT=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" bash -s <<'REMOTE'
cd /workspace/thesis
source .venv/bin/activate
nohup bash -c 'sleep 172800 && /sbin/shutdown -h now' > /tmp/auto_shutdown.log 2>&1 &
SHUTDOWN_PID=$!
nohup bash -c 'cd /workspace/thesis && source .venv/bin/activate && THESIS_RUN_PROFILE=scaleup NLI_DEVICE=cuda python scripts/run_nli_pilot.py && echo "RUN COMPLETE - $(date)" >> nli_run.log' > nli_run.log 2>&1 &
RUN_PID=$!
echo "SHUTDOWN_PID=$SHUTDOWN_PID RUN_PID=$RUN_PID"
sleep 90
echo "=== PROCESSES AT T+90s ==="
ps aux | grep -E "run_nli_pilot|sleep 172800" | grep -v grep
echo "=== LOG (first 40 lines) ==="
head -40 nli_run.log
REMOTE
)
echo "$LAUNCH_OUT"
echo "$LAUNCH_OUT" | grep -q "run_nli_pilot" || stop "Scaleup process not visible at T+90s. Check nli_run.log output above for an error."
echo "$LAUNCH_OUT" | grep -q "0 live fetches" && echo "PREFETCH VERIFIED: 0 live fetches" || echo "[WARN] '0 live fetches' not yet in log — verify manually: grep prefetch nli_run.log"
hr

# ================= STEP 6: Final summary =================
cat <<SUMMARY

=== SCALEUP NLI LAUNCHED ===

Expected workload : 9,992 cells (~6.7x pilot; pilot ETA was ~5h => expect roughly 30-36h)
Hard shutdown     : 48h from now (cost insurance — with checkpointing this is NON-fatal)
Checkpoint        : data/scaleup/gold/nli_scaleup.partial.parquet (every 200 cells + per run)

Monitor progress:
  ssh root@$POD_IP -p $POD_PORT -i ~/.ssh/id_ed25519 "tail -5 /workspace/thesis/nli_run.log"

If the pod dies / shuts down mid-run:
  1. Start a new pod, re-run this script (it re-syncs code + silver).
  2. BEFORE STEP 5, pull the partial back and push it to the new pod:
     scp -P \$POD_PORT -i ~/.ssh/id_ed25519 root@\$POD_IP:/workspace/thesis/data/scaleup/gold/nli_scaleup.partial.parquet .
     (run_nli_pilot.py resumes automatically from it — at most 200 cells are recomputed)

To retrieve gold when complete:
  scp -P \$POD_PORT -i ~/.ssh/id_ed25519 \\
    root@\$POD_IP:/workspace/thesis/data/scaleup/gold/nli_scaleup.parquet \\
    "\$HOME/Desktop/Thesis Data Pilot/data/scaleup/gold/"

Verify locally (expect $EXPECTED_GOLD_ROWS rows; kimi has 249 query-cells, others 250):
  python -c "import pandas as pd; df = pd.read_parquet('data/scaleup/gold/nli_scaleup.parquet'); print(len(df)); print(df.groupby(['engine','run_index']).size())"

Then aggregate locally:
  THESIS_RUN_PROFILE=scaleup python scripts/aggregate_cv_pilot.py   # -> 1,249 cells + gap note

After pulling gold, terminate the pod in the RunPod dashboard to stop charges.
SUMMARY

echo ""
echo "Done at $(date)."
