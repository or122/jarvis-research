#!/bin/bash
# Train Flow's code model in chunks, so the Mac never cooks itself.
#
# Measured on this machine: ~35 minutes of full-speed training, then the CPU
# throttles. Past a few hours it degrades to ~13% of one core and effectively
# stops. Training in bursts with rest between keeps it near full speed the
# whole way, and because train.py resumes from its checkpoint, several short
# sessions produce the same model as one long run.
#
# Usage:  ./train_chunked.sh [iterations_per_chunk] [chunks]
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/gefen/flow/.venv/bin/python"
MODEL="$HERE/model"

PER_CHUNK="${1:-400}"      # ~35 minutes at full speed
CHUNKS="${2:-8}"
REST="${REST:-900}"        # 15 minutes to cool

cd "$MODEL" || exit 1

# The code model uses the 12,288 vocabulary, which is NOT the one the working
# chat model uses. Both must be swapped in together or the model produces
# fluent nonsense (sample.py refuses to load a mismatched pair).
if [ "$(cat ACTIVE 2>/dev/null)" != "code" ]; then
  echo "switching to the code model (12,288 vocab)"
  cp ckpt_code_partial.pt ckpt.pt
  cp data/vocab_code.json data/vocab.json
  echo code > ACTIVE
fi

TARGET=$("$PY" -c "import torch;print(torch.load('ckpt.pt',map_location='cpu')['iter'])")
echo "starting from iteration $TARGET"

for i in $(seq 1 "$CHUNKS"); do
  TARGET=$(( TARGET + PER_CHUNK ))
  echo ""
  echo "=== chunk $i/$CHUNKS  ->  iteration $TARGET  ($(date +%H:%M)) ==="

  FLOW_EVAL_EVERY=100 FLOW_EVAL_ITERS=20 \
    caffeinate -dimsu "$PY" -u train.py "$TARGET" all_ 2>&1 \
    | grep -E --line-buffered "^iter|best val|GATE"

  # Report the CPU's real speed after each chunk, so a thermal slide shows up
  # immediately instead of after an hour of missing log lines.
  "$PY" - <<'PYEOF'
import time, torch
torch.set_num_threads(4)
a = torch.randn(1024, 1024); b = torch.randn(1024, 1024)
a @ b
t = time.time()
for _ in range(40):
    c = a @ b
g = 40 * 2 * 1024 ** 3 / (time.time() - t) / 1e9
print(f"cpu now: {g:.0f} GFLOPS  ({'healthy' if g > 70 else 'THROTTLING'})")
PYEOF

  if [ "$i" -lt "$CHUNKS" ]; then
    echo "resting ${REST}s so the CPU recovers..."
    sleep "$REST"
  fi
done

echo ""
echo "=== done. running the evals ==="
"$PY" eval.py
"$PY" eval_code.py
