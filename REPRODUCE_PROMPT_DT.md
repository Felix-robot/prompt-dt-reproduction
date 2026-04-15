# Prompt-DT 本地复现说明

本仓库位于：

```bash
/mnt/c/Users/yangz/Desktop/GithubWorkspace/Prompt-DT
```

当前运行环境是 WSL2 `Ubuntu-20.04`，conda 环境名为 `prompt-dt`。

## 当前状态

检查日期：2026-04-14

```text
WSL distro: Ubuntu-20.04
Conda env: /home/yang/miniforge3/envs/prompt-dt
Python: 3.8.5
PyTorch: 2.3.1+cu121
PyTorch 使用的 CUDA runtime: 12.1
PyTorch CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
MuJoCo binary: /home/yang/.mujoco/mujoco210
mujoco-py: 2.1.2.14
数据文件: Prompt-DT/data 下 284 个文件，共 4,328,126,973 bytes
```

原项目固定 `torch==1.10.0` 和 `mujoco-py==2.0.2.13`。本机做了两处兼容性调整：

1. RTX 4060 的 compute capability 是 `8.9`。旧的 PyTorch 1.10 CUDA build 可以识别
   GPU，但在实际训练中会触发 `CUDA illegal instruction`。最终可用的 GPU 环境使用
   `torch==2.3.1+cu121`。
2. `mujoco-py==2.0.2.13` 需要旧版 MuJoCo 2.0 授权文件 `mjkey.txt`。本机没有可用的
   `mjkey.txt`，因此使用 MuJoCo 2.1 和 `mujoco-py==2.1.2.14`。

这些是原依赖在当前硬件和系统上的兼容性调整。该环境已经通过 MuJoCo、CUDA、
数据加载和训练 smoke test。

## 已安装内容

Conda/WSL 侧配置：

```text
Miniforge: /home/yang/miniforge3
环境名: prompt-dt
python=3.8.5
patchelf
glew
glfw
mesalib
```

`prompt-dt` 环境中的主要 pip 包：

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

项目内置的本地环境包也已经安装：

```text
metaworld
jacopinpad
mujoco-control-envs
```

因为 `mujoco-control-envs` 不是标准 wheel 结构，conda 环境中设置了 `PYTHONPATH`：

```text
/mnt/c/Users/yangz/Desktop/GithubWorkspace/Prompt-DT/envs/mujoco-control-envs
```

conda 环境中还设置了以下 MuJoCo 相关变量：

```text
MUJOCO_PY_MUJOCO_PATH=/home/yang/.mujoco/mujoco210
LD_LIBRARY_PATH=/home/yang/.mujoco/mujoco210/bin:/home/yang/miniforge3/envs/prompt-dt/lib
CPATH=/home/yang/miniforge3/envs/prompt-dt/include
LIBRARY_PATH=/home/yang/miniforge3/envs/prompt-dt/lib
```

修改这些变量后需要重新激活环境：

```bash
conda deactivate
conda activate prompt-dt
```

## 使用方法

进入 WSL 并激活环境：

```bash
wsl -d Ubuntu-20.04
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prompt-dt
cd /mnt/c/Users/yangz/Desktop/GithubWorkspace/Prompt-DT
```

运行最小 smoke test：

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

不使用 Weights & Biases 时，使用 `--log_to_wandb=`。原项目参数使用 `type=bool`，
因此传入 `False` 或 `0` 反而会被解析成 true。

运行默认实验：

```bash
python pdt_main.py --env cheetah_dir --log_to_wandb=
```

运行本仓库整理好的 demo 脚本：

```bash
bash scripts/run_cheetah_dir_demo.sh
```

该脚本默认使用复现报告中的 2000-iteration 设置。需要快速检查时可以覆盖参数：

```bash
MAX_ITERS=100 NUM_EVAL_EPISODES=3 bash scripts/run_cheetah_dir_demo.sh
```

本机 pip 依赖记录在：

```text
requirements-wsl-rtx40.txt
```

原项目支持的环境：

```text
cheetah_dir
cheetah_vel
ant_dir
ML1-pick-place-v2
```

需要在线 Weights & Biases 记录时：

```bash
wandb login
python pdt_main.py --env cheetah_dir
```

## 数据

README 中给出的官方数据集已经下载并解压。当前布局：

```text
Prompt-DT/
  data/
    ant_dir/
    cheetah_dir/
    cheetah_vel/
    ML1-pick-place-v2/
```

原始 `data.zip` 只在安装时临时保留，解压后已经删除。

## 已完成验证

依赖导入检查：

```bash
conda run -n prompt-dt python -c "import torch, numpy, gym, transformers, wandb, pygame, pytest, matplotlib"
```

MuJoCo 和 CUDA 检查：

```bash
conda run -n prompt-dt python -c "import mujoco_py, torch; print(mujoco_py.__version__); print(torch.cuda.is_available())"
```

环境和数据加载检查：

```bash
conda run -n prompt-dt python -c "from prompt_dt.prompt_utils import get_env_list, load_data_prompt"
```

`cheetah_dir` 的最小训练 smoke run 已完成。GPU demo 也已完成，当前最强记录是
2000 iterations，结果摘要见：

```text
results/cheetah_dir_demo/summary.md
```

## 已知说明

`pip check` 会给出以下预期 warning：

```text
metaworld 0.0.0 requires mujoco-py<2.1,>=2.0, but mujoco-py 2.1.2.14 is installed.
```

这是为了绕开旧版 MuJoCo 2.0 license key 所采用的 MuJoCo 2.1 路线。当前
`cheetah_dir` 环境 reset 和最小训练 smoke run 已经通过。
