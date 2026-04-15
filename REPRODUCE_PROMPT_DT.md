# Prompt-DT Reproduction Notes

This repository was cloned into:

```bash
/mnt/c/Users/yangz/Desktop/GithubWorkspace/Prompt-DT
```

The working runtime is WSL2 `Ubuntu-20.04` with a conda environment named
`prompt-dt`.

## Current Status

Checked on 2026-04-14:

```text
WSL distro: Ubuntu-20.04
Conda env: /home/yang/miniforge3/envs/prompt-dt
Python: 3.8.5
PyTorch: 2.3.1+cu121
CUDA runtime used by PyTorch: 12.1
CUDA available from PyTorch: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
MuJoCo binary used: /home/yang/.mujoco/mujoco210
mujoco-py used: 2.1.2.14
Dataset files: 284 files under Prompt-DT/data, 4,328,126,973 bytes
```

The original repository pins `torch==1.10.0` and `mujoco-py==2.0.2.13`.
Two changes were needed on this machine:

1. RTX 4060 is compute capability `8.9`. The old PyTorch 1.10 CUDA build can
   detect the GPU but failed during non-trivial training with `CUDA illegal
   instruction`. The working GPU environment uses `torch==2.3.1+cu121`.
2. `mujoco-py==2.0.2.13` requires the legacy MuJoCo 2.0 license file
   `mjkey.txt`. No existing `mjkey.txt` was found, so the working setup uses
   MuJoCo 2.1 plus `mujoco-py==2.1.2.14`.

These are documented compatibility differences from the original dependency
pins. The resulting setup passed MuJoCo, CUDA, data-loading, and training smoke
tests.

## Installed Pieces

Conda/WSL pieces installed or configured:

```text
Miniforge: /home/yang/miniforge3
Environment: prompt-dt
python=3.8.5
patchelf
glew
glfw
mesalib
```

Pip packages installed in `prompt-dt`:

```text
Cython==0.29.36
gym==0.18.3
numpy==1.20.3
transformers==4.5.1
wandb==0.9.1
mujoco-py==2.1.2.14
torch==2.3.1+cu121
pygame==2.1.0
pytest==6.2.5
matplotlib==3.5.1
packaging==20.9
pyparsing==2.4.7
imageio==2.9.0
Pillow==8.2.0
```

The bundled local packages were installed into the environment:

```text
metaworld
jacopinpad
mujoco-control-envs
```

Because `mujoco-control-envs` is not packaged cleanly as a normal wheel, the
conda environment sets `PYTHONPATH` to:

```text
/mnt/c/Users/yangz/Desktop/GithubWorkspace/Prompt-DT/envs/mujoco-control-envs
```

The conda environment also stores these MuJoCo variables:

```text
MUJOCO_PY_MUJOCO_PATH=/home/yang/.mujoco/mujoco210
LD_LIBRARY_PATH=/home/yang/.mujoco/mujoco210/bin:/home/yang/miniforge3/envs/prompt-dt/lib
CPATH=/home/yang/miniforge3/envs/prompt-dt/include
LIBRARY_PATH=/home/yang/miniforge3/envs/prompt-dt/lib
```

Reactivate the environment after changing these vars:

```bash
conda deactivate
conda activate prompt-dt
```

## How To Use

Open WSL and activate the environment:

```bash
wsl -d Ubuntu-20.04
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prompt-dt
cd /mnt/c/Users/yangz/Desktop/GithubWorkspace/Prompt-DT
```

Run a small smoke test:

```bash
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
  --log_to_wandb=
```

Use `--log_to_wandb=` when you do not want Weights & Biases. The upstream code
uses `type=bool`, so values like `False` or `0` are parsed as true.

Run the default experiment:

```bash
python pdt_main.py --env cheetah_dir --log_to_wandb=
```

Run the packaged demo script:

```bash
bash scripts/run_cheetah_dir_demo.sh
```

The demo script defaults to the 2000-iteration setting used in the reproduction
report. Override the size for a quicker check if needed:

```bash
MAX_ITERS=100 NUM_EVAL_EPISODES=3 bash scripts/run_cheetah_dir_demo.sh
```

The local pip dependency list is recorded in:

```text
requirements-wsl-rtx40.txt
```

Supported environments:

```text
cheetah_dir
cheetah_vel
ant_dir
ML1-pick-place-v2
```

If you want online Weights & Biases logging:

```bash
wandb login
python pdt_main.py --env cheetah_dir
```

## Data

The README dataset has been downloaded and extracted. Current layout:

```text
Prompt-DT/
  data/
    ant_dir/
    cheetah_dir/
    cheetah_vel/
    ML1-pick-place-v2/
```

The original data zip was only kept in a temporary directory during setup and
has been removed after extraction.

## Verification Already Run

Dependency import check:

```bash
conda run -n prompt-dt python -c "import torch, numpy, gym, transformers, wandb, pygame, pytest, matplotlib"
```

MuJoCo and CUDA check:

```bash
conda run -n prompt-dt python -c "import mujoco_py, torch; print(mujoco_py.__version__); print(torch.cuda.is_available())"
```

Environment and data smoke check:

```bash
conda run -n prompt-dt python -c "from prompt_dt.prompt_utils import get_env_list, load_data_prompt"
```

Tiny training smoke run also completed successfully for `cheetah_dir`.

The `cheetah_dir` GPU demos completed successfully. The strongest completed run
used 2000 iterations and wrote its summary to:

```text
results/cheetah_dir_demo/summary.md
```

## Known Notes

`pip check` reports this expected compatibility warning:

```text
metaworld 0.0.0 requires mujoco-py<2.1,>=2.0, but mujoco-py 2.1.2.14 is installed.
```

This is intentional for the no-license MuJoCo 2.1 route. The `cheetah_dir`
environment reset and the tiny training smoke run passed with this setup.
