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

## 跑的过程中理解了什么

跑之前我对 offline RL 的理解比较抽象。真正把数据加载和训练流程跑起来之后，
最直观的一点是：这里的数据集是固定的。训练时程序从 `data/` 下面读取 `.pkl`
轨迹，agent 并不会一边训练一边和环境交互采样。MuJoCo 环境主要出现在评估阶段，
用来检查当前策略在 `cheetah_dir` 任务里的实际 return。

Prompt-DT 的核心思路也比我一开始想象得更具体。它是在 Decision Transformer
的序列输入里加入一段 demonstration trajectory，让这段轨迹告诉模型当前任务是什么。
在 `cheetah_dir` 里，这个差别很直观：一个任务要求 HalfCheetah 向前跑，另一个任务
要求向后跑。同一个模型看到不同的 prompt trajectory，就应该产生不同方向的动作。

这次复现让我跑通了 Prompt-DT 的完整训练和评估流程，也让我对 in-context decision
making 和 trajectory prompt 作为上下文输入的思路有了更具体的理解。与此同时，我也
清楚这和独立设计复杂 MuJoCo XML、机器人动力学模型，或者微调 LLM/foundation model
不是同一类经验。更准确地说，这次工作是一次基于 Transformer policy model 的离线
强化学习复现。

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

这次实验本地 RTX 4060 就够用了，2000-iteration 的 `cheetah_dir` GPU 训练已经
完整跑通。当前日志没有记录精确的开始和结束时间，所以这里不写具体耗时。

如果后面要做论文级复现，比如 5000 iterations、更多环境和多 seed 统计，云端 GPU
会更合适。本地机器仍然可以继续跑，但长时间实验更容易被断电、断网或日常使用打断。

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
