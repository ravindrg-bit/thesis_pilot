#!/usr/bin/env bash
# RunPod NLI Pilot launcher — faithful to runpod_nli_pilot_dispatch_prompt.md
# Runs from YOUR Mac (uses ~/.ssh/id_ed25519). Double-click in Finder, or:
#   bash "run_runpod_nli_pilot.command"
# Stops at any failure gate. Does NOT push to GitHub. Does NOT touch your key contents.

set -u

# ---- Connection details ----
POD_IP="213.173.105.146"
POD_PORT="32862"
SSH_KEY="$HOME/.ssh/id_ed25519"
REPO="https://github.com/ravindrg-bit/thesis-data-pilot.git"
LOCAL_PROJECT="$HOME/Desktop/Thesis Data Pilot"
EXPECTED_BRONZE=1582

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -i "$SSH_KEY" -p "$POD_PORT")
RSH="ssh -i $SSH_KEY -p $POD_PORT -o StrictHostKeyChecking=accept-new"

stop() { echo ""; echo "==== STOP: $* ===="; echo "Halting. Fix the above, then re-run."; exit 1; }
hr()   { echo "------------------------------------------------------------"; }

echo "RunPod NLI Pilot — starting at $(date)"
hr

# ================= STEP 1: SSH reachability =================
echo "STEP 1 — Verify SSH reachability"
nc -zv -w 5 "$POD_IP" "$POD_PORT" || stop "TCP to $POD_IP:$POD_PORT failed. Likely your network blocks high ports — switch to phone hotspot and re-run."

PROBE=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "echo CONNECTED && hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>&1)
echo "$PROBE"
echo "$PROBE" | grep -q "CONNECTED" || stop "SSH probe did not return CONNECTED. See output above."
hr

# ================= STEP 2: Pod environment setup =================
echo "STEP 2 — Set up pod environment (clone + venv + deps + model + cuda)"
ssh "${SSH_OPTS[@]}" root@"$POD_IP" bash -s <<'REMOTE'
set -e
cd /workspace
if [ ! -d thesis ]; then
  git clone https://github.com/ravindrg-bit/thesis-data-pilot.git thesis 2>&1
fi
cd /workspace/thesis
apt-get update -qq && apt-get install -y -qq tmux wget
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.lock --quiet
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT REACHABLE'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"
mkdir -p data/pilot/bronze data/pilot/silver data/pilot/gold
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; m='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'; AutoTokenizer.from_pretrained(m); AutoModelForSequenceClassification.from_pretrained(m); print('Model cached.')"
sed -i 's/^NLI_DEVICE = "mps"/NLI_DEVICE = "cuda"/' thesis_config.py
echo "DEVICE_LINE: $(grep -n '^NLI_DEVICE' thesis_config.py)"
echo "STRK_COUNT: $(grep -c 'str(k): round' src/nli_attribution.py src/attribution.py | awk -F: '{s+=$2} END{print s}')"
REMOTE
RC=$?
[ $RC -eq 0 ] || stop "Pod setup failed (exit $RC). If you saw a Git auth prompt, STOP — paste a PAT or make the repo public, then re-run. If CUDA NOT REACHABLE, this pod is unusable."

# Re-verify the two gates explicitly
DEV=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "grep '^NLI_DEVICE' /workspace/thesis/thesis_config.py")
echo "$DEV" | grep -q 'NLI_DEVICE = "cuda"' || stop "NLI_DEVICE is not cuda after sed: $DEV"
STRK=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "grep -c 'str(k): round' /workspace/thesis/src/nli_attribution.py /workspace/thesis/src/attribution.py | awk -F: '{s+=\$2} END{print s}'")
[ "$STRK" = "4" ] || stop "Expected exactly 4 'str(k): round' lines on pod, found: $STRK"
echo "Gates OK: CUDA, NLI_DEVICE=cuda, str(k)=4"
hr

# ================= STEP 3: Upload bronze + silver =================
echo "STEP 3 — rsync bronze + silver to pod"
rsync -avz --progress -e "$RSH" \
  "$LOCAL_PROJECT/data/pilot/bronze/" \
  root@"$POD_IP":/workspace/thesis/data/pilot/bronze/ || stop "rsync bronze failed. Retry once (rsync resumes)."

rsync -avz --progress -e "$RSH" \
  "$LOCAL_PROJECT/data/pilot/silver/" \
  root@"$POD_IP":/workspace/thesis/data/pilot/silver/ || stop "rsync silver failed. Retry once (rsync resumes)."

BCOUNT=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "ls /workspace/thesis/data/pilot/bronze | wc -l | tr -d ' '")
[ "$BCOUNT" = "$EXPECTED_BRONZE" ] || stop "Bronze count on pod is $BCOUNT, expected $EXPECTED_BRONZE."
SLIST=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "ls /workspace/thesis/data/pilot/silver")
echo "Silver on pod: $SLIST"
echo "$SLIST" | grep -q "responses_pilot.parquet" && echo "$SLIST" | grep -q "citations_pilot.parquet" || stop "Silver parquet files missing on pod."
echo "Upload verified: bronze=$BCOUNT, silver OK"
hr

# ================= STEP 4: CRITICAL GATE — single-cell write test =================
echo "STEP 4 — single-cell parquet write test (the str(k) gate)"
WRITE_OUT=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" "cd /workspace/thesis && source .venv/bin/activate && python -" <<'PY' 2>&1
import sys
sys.path.insert(0, '.')
from src.env import load_env
load_env()
import pandas as pd
import thesis_config as cfg
from src import nli_attribution as nli

responses = pd.read_parquet(cfg.SILVER / f'responses_{cfg.RUN_PROFILE}.parquet')
citations = pd.read_parquet(cfg.SILVER / f'citations_{cfg.RUN_PROFILE}.parquet')
cell = responses[(responses.engine == 'claude')].iloc[0]
qid, run = cell['query_id'], int(cell['run_index'])
answer = cell['answer_text']
srcs = (citations[(citations.engine == 'claude') & (citations.query_id == qid) & (citations.run_index == run)]
        .sort_values('position')[['position', 'url']].to_dict('records'))
res = nli.attribute_cell_nli(answer, srcs, {})
res.update({'engine': 'claude', 'query_id': qid, 'run_index': run})
df = pd.DataFrame([res])
test_path = cfg.GOLD / 'TEST_pilot_write.parquet'
df.to_parquet(test_path, index=False)
df2 = pd.read_parquet(test_path)
keys = list(df2.iloc[0]['pawc_by_source_position'].keys())
print(f'WRITE_OK={len(df2)==1} | KEYS_ALL_STRINGS={all(isinstance(k, str) for k in keys)}')
PY
)
echo "$WRITE_OUT"
echo "$WRITE_OUT" | grep -q "WRITE_OK=True | KEYS_ALL_STRINGS=True" || stop "Write test did not pass. DO NOT launch. Traceback is above."
ssh "${SSH_OPTS[@]}" root@"$POD_IP" "rm -f /workspace/thesis/data/pilot/gold/TEST_pilot_write.parquet"
echo "Write test PASSED and test artifact cleaned."
hr

# ================= STEP 5: Launch pilot + hard shutdown insurance =================
echo "STEP 5 — Launch pilot in background with 6h hard-shutdown timer"
LAUNCH_OUT=$(ssh "${SSH_OPTS[@]}" root@"$POD_IP" bash -s <<'REMOTE'
cd /workspace/thesis
source .venv/bin/activate
nohup bash -c 'sleep 21600 && /sbin/shutdown -h now' > /tmp/auto_shutdown.log 2>&1 &
SHUTDOWN_PID=$!
nohup bash -c 'cd /workspace/thesis && source .venv/bin/activate && NLI_DEVICE=cuda NLI_BATCH=32 python scripts/run_nli_pilot.py && echo "RUN COMPLETE - $(date)" >> nli_run.log' > nli_run.log 2>&1 &
PILOT_PID=$!
echo "SHUTDOWN_PID=$SHUTDOWN_PID PILOT_PID=$PILOT_PID"
sleep 60
echo "=== PROCESSES AT T+60s ==="
ps aux | grep -E "run_nli_pilot|sleep 21600" | grep -v grep
echo "=== LOG (first 30 lines) ==="
head -30 nli_run.log
REMOTE
)
echo "$LAUNCH_OUT"
echo "$LAUNCH_OUT" | grep -q "run_nli_pilot" || stop "Pilot process not visible at T+60s. Check nli_run.log output above for an error."

PILOT_PID=$(echo "$LAUNCH_OUT" | grep -oE "PILOT_PID=[0-9]+" | head -1 | cut -d= -f2)
SHUTDOWN_PID=$(echo "$LAUNCH_OUT" | grep -oE "SHUTDOWN_PID=[0-9]+" | head -1 | cut -d= -f2)
hr

# ================= STEP 6: Final summary =================
cat <<SUMMARY

=== PILOT LAUNCHED SUCCESSFULLY ===

Pod: pwf7qsycoeeo7m (upper_aqua_crayfish)
SSH for reconnection tomorrow:
  ssh root@$POD_IP -p $POD_PORT -i ~/.ssh/id_ed25519
  (IP/port may change after pod restart — get fresh from RunPod Connect dialog)

Pilot PID: $PILOT_PID
Shutdown timer PID: $SHUTDOWN_PID
Expected completion: ~5 hours from now
Hard shutdown: 6 hours from now (will halt pod regardless of pilot state)

To retrieve gold parquet tomorrow morning:
  POD_IP=<new_ip_from_connect_dialog>
  POD_PORT=<new_port_from_connect_dialog>
  mkdir -p "\$HOME/Desktop/Thesis Data Pilot/data/pilot/gold_runpod"
  scp -P \$POD_PORT -i ~/.ssh/id_ed25519 \\
    root@\$POD_IP:/workspace/thesis/data/pilot/gold/pilot_nli_pilot.parquet \\
    "\$HOME/Desktop/Thesis Data Pilot/data/pilot/gold_runpod/"

To verify locally:
  python -c "import pandas as pd; df = pd.read_parquet('\$HOME/Desktop/Thesis Data Pilot/data/pilot/gold_runpod/pilot_nli_pilot.parquet'); print(f'Rows: {len(df)}'); print(df.groupby(['engine','run_index']).size())"

Expect ~1500 rows, ~100 per (engine, run_index) cell.

After pulling gold, terminate the pod in RunPod dashboard (trash icon) to stop storage charges.
SUMMARY

echo ""
echo "Done at $(date). You can close the laptop."
