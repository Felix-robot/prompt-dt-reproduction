# Prompt-DT Reproduction Report

## Reproduction Scope

I set up the released Prompt-DT codebase locally on WSL2 Ubuntu 20.04 and
verified the training pipeline on `cheetah_dir` with an RTX 4060 Laptop GPU.

This report does not claim to reproduce every table in the paper. It records an
end-to-end local run covering the released code, official dataset, MuJoCo
environment, Prompt-DT model, training loop, and evaluation loop.

## Environment

Run date: 2026-04-14

| Item | Value |
| --- | --- |
| OS | WSL2 Ubuntu 20.04 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Python | 3.8.5 |
| PyTorch | 2.3.1+cu121 |
| CUDA visible to PyTorch | yes |
| MuJoCo | 2.1 |
| mujoco-py | 2.1.2.14 |
| Dataset | official Prompt-DT Google Drive data |

Two dependency changes were necessary:

- The original `torch==1.10.0` CUDA stack detected the RTX 4060, but training
  failed with `CUDA illegal instruction`. The RTX 4060 is compute capability
  8.9, so I moved to `torch==2.3.1+cu121`.
- The original `mujoco-py==2.0.2.13` path requires a legacy MuJoCo 2.0
  `mjkey.txt`. No key was available locally, so I used MuJoCo 2.1 with
  `mujoco-py==2.1.2.14`.

These are engineering compatibility changes, not algorithm changes.

## Background Concepts Reviewed

- Offline RL uses fixed trajectory datasets. Here, training reads `.pkl`
  trajectories from `data/<env>/` instead of collecting new online data.
- Decision Transformer frames policy learning as sequence modeling over return,
  state, action, and timestep tokens.
- Prompt-DT adds task context through prompt trajectories. The prompt is not a
  text prompt; it is a short trajectory context used for in-context policy
  adaptation.
- `cheetah_dir` is a two-task benchmark: one task runs forward, the other runs
  backward. The same model is trained and evaluated across both.
- A large part of the reproduction work was dependency and systems work:
  MuJoCo, gym, transformers, CUDA compatibility, and native library paths.

## Code Map

| Path | Role |
| --- | --- |
| `pdt_main.py` | experiment entry point; builds envs, loads data, trains, evaluates |
| `prompt_dt/prompt_decision_transformer.py` | Prompt-DT model |
| `prompt_dt/prompt_seq_trainer.py` | training and evaluation loops |
| `prompt_dt/prompt_utils.py` | dataset loading, prompt batching, env construction |
| `config/` | train/test task splits |
| `data/` | official offline datasets, ignored by Git |
| `envs/` | bundled MuJoCo and MetaWorld environments |
| `scripts/run_cheetah_dir_demo.sh` | reproducible local demo command |

## Dataset Check

The official data archive was downloaded and extracted locally.

| Dataset | Trajectories | Timesteps | Dataset Return Mean |
| --- | ---: | ---: | ---: |
| `cheetah_dir-0` | 999 | 199800 | 666.58 |
| `cheetah_dir-1` | 999 | 199800 | 1134.30 |

Total local dataset size:

```text
284 files
4,328,126,973 bytes
```

The data directory is intentionally not committed.

## Experiment

Main completed run:

```bash
python -u pdt_main.py --env cheetah_dir --max_iters 2000 --num_steps_per_iter 10 --num_eval_episodes 10 --device cuda --batch_size 16 --embed_dim 128 --n_layer 3 --n_head 1 --K 20 --prompt-length 5 --train_eval_interval 500 --test_eval_interval 500 --save-interval 999999 --log_to_wandb=
```

Run log:

```text
results/cheetah_dir_demo/run_20260414_215515/train.log
```

Test evaluation:

| Iteration | cheetah_dir-0 Return | cheetah_dir-1 Return | Action Error |
| ---: | ---: | ---: | ---: |
| 1 | 4.72 | -28.84 | 1.0908 |
| 501 | 399.43 | 657.00 | 0.0711 |
| 1001 | 593.91 | 745.86 | 0.0496 |
| 1501 | 628.15 | 934.36 | 0.0428 |

Train evaluation:

| Iteration | cheetah_dir-0 Return | cheetah_dir-1 Return | Action Error |
| ---: | ---: | ---: | ---: |
| 1 | 8.07 | -28.91 | 1.0908 |
| 501 | 435.52 | 542.31 | 0.0711 |
| 1001 | 588.37 | 814.05 | 0.0496 |
| 1501 | 644.43 | 1086.59 | 0.0428 |

The run gives a clear local reproduction signal:

- action prediction error dropped from `1.0908` to `0.0428`
- test `cheetah_dir-0` improved from `4.72` to `628.15`
- test `cheetah_dir-1` improved from `-28.84` to `934.36`
- train returns approached the scale of the expert datasets

## Local GPU vs Cloud Server

A cloud server is not required for the current reproduction evidence. The local
RTX 4060 completed the 2000-iteration `cheetah_dir` GPU run.

Cloud becomes useful if the target changes from "local reproduction proof" to
"paper-level reproduction":

| Goal | Local RTX 4060 | Cloud GPU |
| --- | --- | --- |
| smoke test | enough | unnecessary |
| `cheetah_dir` 2000 iterations | enough | unnecessary |
| `cheetah_dir` 5000 iterations | likely enough, longer wait | optional |
| `cheetah_vel` single run | likely enough | optional |
| `ant_dir` or MetaWorld long runs | possible but slower | useful |
| multi-seed paper table | inconvenient | recommended |

For a portfolio-style reproduction record, this local run is enough to show the
setup, compatibility fixes, GPU training, and evaluation output. For a
paper-style comparison, cloud compute would mainly save time and reduce
interruption risk.

## Limitations

- This is a single local setup, not a multi-machine reproduction.
- The strongest completed run is 2000 iterations, not the default 5000.
- Only `cheetah_dir` has been benchmarked so far.
- No multi-seed statistics are reported.
- MuJoCo 2.1 was used instead of licensed MuJoCo 2.0.

## Next Steps

The next highest-value steps are:

1. Run `cheetah_dir` for 5000 iterations.
2. Run `cheetah_vel` with the same reporting format.
3. Add seed control and run at least 3 seeds.
4. Compare final numbers against the paper or official reported results.
