# Prompt-DT 复现报告

## 复现范围

我在 WSL2 Ubuntu 20.04 上搭建了 Prompt-DT 的本地运行环境，并使用
RTX 4060 Laptop GPU 在 `cheetah_dir` 环境上验证了训练流程。

这份报告不声称复现了论文中的所有实验表格。它记录的是一次端到端本地复现：
使用原项目代码、官方数据集、MuJoCo 环境、Prompt-DT 模型、训练循环和评估循环，
完成了可检查的 GPU 训练结果。

## 环境配置

运行日期：2026-04-14

| 项目 | 配置 |
| --- | --- |
| 操作系统 | WSL2 Ubuntu 20.04 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Python | 3.8.5 |
| PyTorch | 2.3.1+cu121 |
| PyTorch 是否可见 CUDA | 是 |
| MuJoCo | 2.1 |
| mujoco-py | 2.1.2.14 |
| 数据集 | Prompt-DT 官方 Google Drive 数据 |

本机环境相对原始依赖做了两处兼容性调整：

- 原项目固定的 `torch==1.10.0` CUDA 版本可以识别 RTX 4060，但在实际训练中触发
  `CUDA illegal instruction`。RTX 4060 的 compute capability 是 `8.9`，因此最终使用
  `torch==2.3.1+cu121`。
- 原项目固定的 `mujoco-py==2.0.2.13` 依赖旧版 MuJoCo 2.0 授权文件 `mjkey.txt`。
  本机没有可用的 `mjkey.txt`，因此使用 MuJoCo 2.1 和 `mujoco-py==2.1.2.14`。

这些是工程兼容性调整，不涉及算法逻辑修改。

## 复现中梳理的基础知识

- Offline RL 使用固定轨迹数据集训练策略。本项目从 `data/<env>/` 读取 `.pkl`
  轨迹，而不是在线和环境交互采样。
- Decision Transformer 将策略学习转成序列建模问题，输入序列包含 return、state、
  action 和 timestep。
- Prompt-DT 在 Decision Transformer 基础上加入 prompt trajectory。这里的 prompt
  不是自然语言提示词，而是一段短轨迹上下文，用于让模型根据上下文适应不同任务。
- `cheetah_dir` 是一个双任务环境：一个任务要求 HalfCheetah 向前跑，另一个任务要求
  向后跑。同一个模型需要在两个方向任务上训练和评估。
- 这次复现的主要难点不只在模型代码，而在 MuJoCo、gym、transformers、CUDA 版本和
  native library path 的兼容性处理。

## 能力边界

这次工作可以支撑的表述：

- 具备 MuJoCo/mujoco-py 环境配置、Gym 风格 MuJoCo 环境运行、依赖排错和训练验证经验。
- 具备 PyTorch GPU 训练、CUDA 兼容性排查、离线强化学习数据加载和实验记录经验。
- 具备 Transformer policy model 或 Decision Transformer 类序列模型的训练经验。
- 理解 Prompt-DT 中基于轨迹上下文的 in-context policy adaptation。

不建议过度包装的表述：

- 不建议直接写成“熟练掌握 MuJoCo 物理建模”。本项目证明的是能配置、运行和调试
  MuJoCo 环境，不等同于已经能独立设计复杂 MuJoCo XML 或机器人动力学模型。
- 不建议直接写成“大模型/foundation model 微调经验”。Prompt-DT 使用 Transformer
  架构和 prompt/context 思想，但它不是 LLM，也不是通常意义上的 foundation model。
  更准确的说法是“基于 Transformer 的策略模型训练经验”或“Decision Transformer
  方向的离线强化学习复现经验”。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `pdt_main.py` | 实验入口；构建环境、加载数据、训练并评估 |
| `prompt_dt/prompt_decision_transformer.py` | Prompt-DT 模型实现 |
| `prompt_dt/prompt_seq_trainer.py` | 训练循环和评估循环 |
| `prompt_dt/prompt_utils.py` | 数据加载、prompt batch 构造、环境创建 |
| `config/` | 训练和测试任务划分 |
| `data/` | 官方离线数据集；本地保留，不提交到 Git |
| `envs/` | 项目内置的 MuJoCo 和 MetaWorld 环境 |
| `scripts/run_cheetah_dir_demo.sh` | 可复现实验命令脚本 |

## 数据集检查

官方数据集已经下载并解压到本地。

| 数据集 | 轨迹数 | Timesteps | 数据集平均 return |
| --- | ---: | ---: | ---: |
| `cheetah_dir-0` | 999 | 199800 | 666.58 |
| `cheetah_dir-1` | 999 | 199800 | 1134.30 |

本地数据总量：

```text
284 files
4,328,126,973 bytes
```

`data/` 目录只在本地使用，已经通过 `.gitignore` 排除。

## 实验设置

主要完成的训练命令：

```bash
python -u pdt_main.py --env cheetah_dir --max_iters 2000 --num_steps_per_iter 10 --num_eval_episodes 10 --device cuda --batch_size 16 --embed_dim 128 --n_layer 3 --n_head 1 --K 20 --prompt-length 5 --train_eval_interval 500 --test_eval_interval 500 --save-interval 999999 --log_to_wandb=
```

日志位置：

```text
results/cheetah_dir_demo/run_20260414_215515/train.log
```

测试集评估：

| Iteration | cheetah_dir-0 Return | cheetah_dir-1 Return | Action Error |
| ---: | ---: | ---: | ---: |
| 1 | 4.72 | -28.84 | 1.0908 |
| 501 | 399.43 | 657.00 | 0.0711 |
| 1001 | 593.91 | 745.86 | 0.0496 |
| 1501 | 628.15 | 934.36 | 0.0428 |

训练集评估：

| Iteration | cheetah_dir-0 Return | cheetah_dir-1 Return | Action Error |
| ---: | ---: | ---: | ---: |
| 1 | 8.07 | -28.91 | 1.0908 |
| 501 | 435.52 | 542.31 | 0.0711 |
| 1001 | 588.37 | 814.05 | 0.0496 |
| 1501 | 644.43 | 1086.59 | 0.0428 |

这次运行给出了明确的本地复现信号：

- action prediction error 从 `1.0908` 降到 `0.0428`
- 测试环境 `cheetah_dir-0` return 从 `4.72` 提升到 `628.15`
- 测试环境 `cheetah_dir-1` return 从 `-28.84` 提升到 `934.36`
- 训练环境 return 接近 expert 数据集的量级

## 本地 GPU 与云端服务器

当前复现证据不需要云端服务器。本地 RTX 4060 已经完成了 2000-iteration
`cheetah_dir` GPU 训练。

如果目标从“本地复现证明”变成“论文级完整复现”，云端会更有价值：

| 目标 | 本地 RTX 4060 | 云端 GPU |
| --- | --- | --- |
| smoke test | 足够 | 不需要 |
| `cheetah_dir` 2000 iterations | 足够 | 不需要 |
| `cheetah_dir` 5000 iterations | 大概率可以，只是等待更久 | 可选 |
| `cheetah_vel` 单次实验 | 大概率可以 | 可选 |
| `ant_dir` 或 MetaWorld 长时间实验 | 可以尝试，但耗时更长 | 有帮助 |
| 多 seed 论文级表格 | 不方便 | 建议使用 |

对于作品集式复现记录，本地结果足以展示环境搭建、兼容性排错、GPU 训练和结果记录。
如果要做论文级对比，云端主要用于节省时间并降低本地中断风险。

## 局限性

- 这是单机本地复现，不是多机器复现。
- 当前最强完成 run 是 2000 iterations，不是原始默认的 5000 iterations。
- 目前只完成了 `cheetah_dir` 环境的记录。
- 暂未报告多 seed 统计。
- 使用的是 MuJoCo 2.1，而不是需要旧 license key 的 MuJoCo 2.0。

## 下一步

后续最有价值的扩展是：

1. 将 `cheetah_dir` 跑到 5000 iterations。
2. 用同样格式跑 `cheetah_vel`。
3. 加入 seed 控制并至少跑 3 个 seed。
4. 与论文或官方报告结果做数值对比。
