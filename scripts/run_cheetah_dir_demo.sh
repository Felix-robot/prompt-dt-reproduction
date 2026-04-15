#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi

set +u
conda activate prompt-dt
set -u

MAX_ITERS="${MAX_ITERS:-2000}"
NUM_STEPS_PER_ITER="${NUM_STEPS_PER_ITER:-10}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TEST_EVAL_INTERVAL="${TEST_EVAL_INTERVAL:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-999999}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EMBED_DIM="${EMBED_DIM:-128}"
N_LAYER="${N_LAYER:-3}"
N_HEAD="${N_HEAD:-1}"
K_CONTEXT="${K_CONTEXT:-20}"
PROMPT_LENGTH="${PROMPT_LENGTH:-5}"

RUN_DIR="results/cheetah_dir_demo/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

COMMAND=(
  python
  -u
  pdt_main.py
  --env cheetah_dir
  --max_iters "$MAX_ITERS"
  --num_steps_per_iter "$NUM_STEPS_PER_ITER"
  --num_eval_episodes "$NUM_EVAL_EPISODES"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --embed_dim "$EMBED_DIM"
  --n_layer "$N_LAYER"
  --n_head "$N_HEAD"
  --K "$K_CONTEXT"
  --prompt-length "$PROMPT_LENGTH"
  --train_eval_interval "$TRAIN_EVAL_INTERVAL"
  --test_eval_interval "$TEST_EVAL_INTERVAL"
  --save-interval "$SAVE_INTERVAL"
  --log_to_wandb=
)

{
  echo "root=$ROOT_DIR"
  echo "date=$(date -Is)"
  echo "max_iters=$MAX_ITERS"
  echo "num_steps_per_iter=$NUM_STEPS_PER_ITER"
  echo "num_eval_episodes=$NUM_EVAL_EPISODES"
  echo "device=$DEVICE"
  echo "batch_size=$BATCH_SIZE"
  echo "embed_dim=$EMBED_DIM"
  echo "n_layer=$N_LAYER"
  echo "n_head=$N_HEAD"
  echo "K=$K_CONTEXT"
  echo "prompt_length=$PROMPT_LENGTH"
  python -V
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
  python -c "import mujoco_py; print('mujoco_py', mujoco_py.__version__)"
} | tee "$RUN_DIR/environment.txt"

printf '%q ' "${COMMAND[@]}" | tee "$RUN_DIR/command.txt"
echo | tee -a "$RUN_DIR/command.txt"

"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/train.log"

echo "status=completed" | tee "$RUN_DIR/status.txt"
