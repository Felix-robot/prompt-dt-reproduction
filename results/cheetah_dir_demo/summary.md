# Cheetah 方向任务 Demo 结果

本目录记录了两次本地 `cheetah_dir` 运行。运行环境是 WSL2 Ubuntu 20.04，
GPU 为 RTX 4060 Laptop GPU。

## 环境

```text
OS: WSL2 Ubuntu 20.04
Python: 3.8.5
PyTorch: 2.3.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
MuJoCo: 2.1
mujoco-py: 2.1.2.14
```

数据集：

```text
cheetah_dir-0: 999 trajectories, 199800 timesteps, dataset return mean 666.58
cheetah_dir-1: 999 trajectories, 199800 timesteps, dataset return mean 1134.30
```

## 运行记录

| 运行 | 目录 | Iterations | Eval Episodes | Device | 状态 |
| --- | --- | ---: | ---: | --- | --- |
| 500-iter demo | `run_20260414_213804` | 500 | 5 | cuda | 完成 |
| 2000-iter demo | `run_20260414_215515` | 2000 | 10 | cuda | 完成 |

500-iter 命令：

```bash
python -u pdt_main.py --env cheetah_dir --max_iters 500 --num_steps_per_iter 10 --num_eval_episodes 5 --device cuda --batch_size 16 --embed_dim 128 --n_layer 3 --n_head 1 --K 20 --prompt-length 5 --train_eval_interval 250 --test_eval_interval 250 --save-interval 999999 --log_to_wandb=
```

2000-iter 命令：

```bash
python -u pdt_main.py --env cheetah_dir --max_iters 2000 --num_steps_per_iter 10 --num_eval_episodes 10 --device cuda --batch_size 16 --embed_dim 128 --n_layer 3 --n_head 1 --K 20 --prompt-length 5 --train_eval_interval 500 --test_eval_interval 500 --save-interval 999999 --log_to_wandb=
```

## 测试集结果

| 运行 | Iteration | cheetah_dir-0 Return | cheetah_dir-1 Return | Action Error |
| --- | ---: | ---: | ---: | ---: |
| 500-iter | 1 | -11.61 | -14.26 | 1.0829 |
| 500-iter | 251 | 354.17 | 346.29 | 0.1257 |
| 2000-iter | 1 | 4.72 | -28.84 | 1.0908 |
| 2000-iter | 501 | 399.43 | 657.00 | 0.0711 |
| 2000-iter | 1001 | 593.91 | 745.86 | 0.0496 |
| 2000-iter | 1501 | 628.15 | 934.36 | 0.0428 |

## 训练集结果

| 运行 | Iteration | cheetah_dir-0 Return | cheetah_dir-1 Return | Action Error |
| --- | ---: | ---: | ---: | ---: |
| 500-iter | 1 | -14.16 | -17.50 | 1.0829 |
| 500-iter | 251 | 309.27 | 303.81 | 0.1257 |
| 2000-iter | 1 | 8.07 | -28.91 | 1.0908 |
| 2000-iter | 501 | 435.52 | 542.31 | 0.0711 |
| 2000-iter | 1001 | 588.37 | 814.05 | 0.0496 |
| 2000-iter | 1501 | 644.43 | 1086.59 | 0.0428 |

## 结果解读

500-iteration 运行用于验证 Prompt-DT 的本地训练流程能够完整跑通。2000-iteration
运行显示出更明显的学习趋势：

- 测试环境 `cheetah_dir-0` 从 500-iter demo 点的 `354.17` 提升到 iteration 1501 的
  `628.15`
- 测试环境 `cheetah_dir-1` 从 `346.29` 提升到 `934.36`
- action prediction error 从约 `0.126` 降到约 `0.043`

这些结果还不是完整论文级复现。它们是一组本地单 seed 的 `cheetah_dir` 复现记录，
能够证明训练循环正常工作、两个方向任务上的 return 都在提升，并且更长训练带来了
更好的结果。
