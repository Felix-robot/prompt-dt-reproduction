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

RUN_DIR="results/smoke/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

{
  echo "root=$ROOT_DIR"
  echo "date=$(date -Is)"
  python -V
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  python -c "import mujoco_py; print('mujoco_py', mujoco_py.__version__)"
} | tee "$RUN_DIR/environment.txt"

python -c "from types import SimpleNamespace; from prompt_dt.prompt_utils import get_env_list, load_data_prompt, process_info; import os; args=SimpleNamespace(env='cheetah_dir'); names=['cheetah_dir-0']; info, envs=get_env_list(names, os.path.join(os.getcwd(), 'config'), 'cuda'); trajs, prompts=load_data_prompt(names, os.path.join(os.getcwd(), 'data'), 'expert', 'expert', args); info=process_info(names, trajs, info, 'normal', 'expert', 1.0, {'average_state_mean': False}); obs=envs[0].reset(); print('smoke ok', len(trajs[0]), len(prompts[0]), obs.shape, info[names[0]]['state_dim']); envs[0].close()" \
  2>&1 | tee "$RUN_DIR/smoke.log"

python pdt_main.py \
  --env cheetah_dir \
  --max_iters 1 \
  --num_steps_per_iter 1 \
  --num_eval_episodes 1 \
  --batch_size 2 \
  --embed_dim 32 \
  --n_layer 1 \
  --n_head 1 \
  --K 5 \
  --prompt-length 2 \
  --train_eval_interval 999 \
  --test_eval_interval 999 \
  --save-interval 999 \
  --log_to_wandb= \
  2>&1 | tee "$RUN_DIR/training_smoke.log"

echo "status=completed" | tee "$RUN_DIR/status.txt"
