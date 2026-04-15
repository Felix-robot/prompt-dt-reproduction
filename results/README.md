# 复现实验结果

本目录保存本地复现实验产生的小型文本结果。

数据集、Weights & Biases 目录和模型 checkpoint 等大文件已经通过 `.gitignore`
排除，不会提交到仓库。

当前记录：

```text
smoke/
  导入依赖、创建 MuJoCo 环境、加载数据和最小训练流程的检查结果。
  该目录只会在运行 smoke 脚本后生成。

cheetah_dir_demo/
  `cheetah_dir` benchmark 的本地 demo 运行记录。
```
