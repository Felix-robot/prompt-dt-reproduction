# Cheetah Direction Demo Results

This directory records two local `cheetah_dir` runs on WSL2 Ubuntu 20.04 with
an RTX 4060 Laptop GPU.

## Environment

```text
OS: WSL2 Ubuntu 20.04
Python: 3.8.5
PyTorch: 2.3.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
MuJoCo: 2.1
mujoco-py: 2.1.2.14
```

Dataset:

```text
cheetah_dir-0: 999 trajectories, 199800 timesteps, dataset return mean 666.58
cheetah_dir-1: 999 trajectories, 199800 timesteps, dataset return mean 1134.30
```

## Runs

| Run | Directory | Iters | Eval Episodes | Device | Status |
| --- | --- | ---: | ---: | --- | --- |
| 500-iter demo | `run_20260414_213804` | 500 | 5 | cuda | completed |
| 2000-iter demo | `run_20260414_215515` | 2000 | 10 | cuda | completed |

500-iter command:

```bash
python -u pdt_main.py --env cheetah_dir --max_iters 500 --num_steps_per_iter 10 --num_eval_episodes 5 --device cuda --batch_size 16 --embed_dim 128 --n_layer 3 --n_head 1 --K 20 --prompt-length 5 --train_eval_interval 250 --test_eval_interval 250 --save-interval 999999 --log_to_wandb=
```

2000-iter command:

```bash
python -u pdt_main.py --env cheetah_dir --max_iters 2000 --num_steps_per_iter 10 --num_eval_episodes 10 --device cuda --batch_size 16 --embed_dim 128 --n_layer 3 --n_head 1 --K 20 --prompt-length 5 --train_eval_interval 500 --test_eval_interval 500 --save-interval 999999 --log_to_wandb=
```

## Test Results

| Run | Iteration | cheetah_dir-0 Return | cheetah_dir-1 Return | Action Error |
| --- | ---: | ---: | ---: | ---: |
| 500-iter | 1 | -11.61 | -14.26 | 1.0829 |
| 500-iter | 251 | 354.17 | 346.29 | 0.1257 |
| 2000-iter | 1 | 4.72 | -28.84 | 1.0908 |
| 2000-iter | 501 | 399.43 | 657.00 | 0.0711 |
| 2000-iter | 1001 | 593.91 | 745.86 | 0.0496 |
| 2000-iter | 1501 | 628.15 | 934.36 | 0.0428 |

## Train Results

| Run | Iteration | cheetah_dir-0 Return | cheetah_dir-1 Return | Action Error |
| --- | ---: | ---: | ---: | ---: |
| 500-iter | 1 | -14.16 | -17.50 | 1.0829 |
| 500-iter | 251 | 309.27 | 303.81 | 0.1257 |
| 2000-iter | 1 | 8.07 | -28.91 | 1.0908 |
| 2000-iter | 501 | 435.52 | 542.31 | 0.0711 |
| 2000-iter | 1001 | 588.37 | 814.05 | 0.0496 |
| 2000-iter | 1501 | 644.43 | 1086.59 | 0.0428 |

## Interpretation

The 500-iter run was sufficient to verify that the full Prompt-DT training
pipeline works locally. The 2000-iter run shows substantial additional learning:

- test `cheetah_dir-0` improved from `354.17` at the 500-iter demo point to
  `628.15` at iteration 1501
- test `cheetah_dir-1` improved from `346.29` to `934.36`
- action prediction error dropped from about `0.126` to about `0.043`

The results are still not a full paper-level reproduction. They are a local
single-seed reproduction demonstration on `cheetah_dir`, with evidence that the
training loop is working, returns improve on both direction tasks, and longer
training improves the result.
