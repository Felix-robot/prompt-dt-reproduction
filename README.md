# Prompting Decision Transformer for Few-Shot Policy Generalization

Prompt-DT 官方代码仓库。[[项目主页]](https://mxu34.github.io/PromptDT/)[[论文]](https://mxu34.github.io/PromptDT/pdf/PromptDT.pdf)

## 本地复现说明

这个工作副本增加了一组本地复现记录。复现环境为 WSL2 Ubuntu 20.04 和 RTX 4060
Laptop GPU。

- 环境说明：[`REPRODUCE_PROMPT_DT.md`](REPRODUCE_PROMPT_DT.md)
- 复现报告：[`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md)
- Demo 结果摘要：[`results/cheetah_dir_demo/summary.md`](results/cheetah_dir_demo/summary.md)
- 可运行脚本：[`scripts/run_smoke.sh`](scripts/run_smoke.sh) 和
  [`scripts/run_cheetah_dir_demo.sh`](scripts/run_cheetah_dir_demo.sh)

原项目依赖仍保留在 `requirements.txt`。本机 RTX 40 系列环境使用
`torch==2.3.1+cu121` 和 `mujoco-py==2.1.2.14`；原因见复现报告中的兼容性说明。
当前已经完成一次 2000-iteration 的 `cheetah_dir` GPU 运行，日志保存在
`results/cheetah_dir_demo/`。

Prompt-DT 架构图：

![Teaser](https://github.com/mxu34/mxu34.github.io/blob/master/PromptDT/img/PromptDT.png)

## 安装

原项目在 Ubuntu 20.04 上测试。

- 建议使用 Anaconda 或 Miniforge 创建虚拟环境。

```bash
conda create --name prompt-dt python=3.8.5
conda activate prompt-dt
```

- 实验需要 MuJoCo 和 mujoco-py。安装方式参考
  [mujoco-py repo](https://github.com/openai/mujoco-py)。

- 安装依赖和环境：

```bash
# 安装依赖
pip install -r requirements.txt

# 安装项目环境
./install_envs.sh
```

- 原项目使用 [wandb](https://wandb.ai/site?utm_source=google&utm_medium=cpc&utm_campaign=Performance-Max&utm_content=site&gclid=CjwKCAjwlqOXBhBqEiwA-hhitGcG5-wtdqoNgKyWdNpsRedsbEYyK9NeKcu8RFym6h8IatTjLFYliBoCbikQAvD_BwE)
  记录实验。需要在线记录时，可参考
  [wandb quickstart doc](https://docs.wandb.ai/quickstart) 创建账号。

## 下载数据集

- 原项目通过这个 [Google Drive 链接](https://drive.google.com/drive/folders/1six767uD8yfdgoGIYW86sJY-fmMdYq7e?usp=sharing)
  提供示例数据集。
- 下载 `data` 文件夹。

```bash
wget -O data.zip 'https://drive.google.com/uc?export=download&id=1rZufm-XRq1Ig-56DejkQUX1si_WzCGBe&confirm=True'
unzip data.zip
rm data.zip
```

- 目录结构应为：

```text
.
|-- config
|-- data
|   |-- ant_dir
|   |-- cheetah_dir
|   |-- cheetah_vel
|   `-- ML1-pick-place-v2
|-- envs
|-- prompt_dt
`-- ...
```

## 运行实验

```bash
# Prompt-DT
python pdt_main.py --env cheetah_dir # choices: ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']

# Prompt-MT-BC
python pdt_main.py --no-rtg --no-r

# MT-ORL
python pdt_main.py --no-prompt

# MT-BC-Finetune
python pdt_main.py --no-prompt --no-rtg --no-r --finetune
```

## 致谢

Prompt-DT 代码基于 [decision-transformer](https://github.com/kzl/decision-transformer)。
环境部分基于 [macaw](https://github.com/eric-mitchell/macaw)、
[rand_param_envs](https://github.com/dennisl88/rand_param_envs) 和
[metaworld](https://github.com/rlworkgroup/metaworld) 等项目。

## 引用

如果本项目对研究有帮助，请引用原论文。

```bibtex
@inproceedings{xu2022prompting,
  title={Prompting Decision Transformer for Few-Shot Policy Generalization},
  author={Xu, Mengdi and Shen, Yikang and Zhang, Shun and Lu, Yuchen and Zhao, Ding and Tenenbaum, Joshua and Gan, Chuang},
  booktitle={International Conference on Machine Learning},
  pages={24631--24645},
  year={2022},
  organization={PMLR}
}
```

## 贡献

关于原项目的改进建议可发送邮件至 mengdixu@andrew.cmu.edu。
