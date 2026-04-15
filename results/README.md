# Reproduction Results

This directory stores small, text-based outputs from local reproduction runs.

Large generated artifacts such as datasets, Weights & Biases directories, and
model checkpoints are intentionally ignored by `.gitignore`.

Current recorded runs:

```text
smoke/
  One-step checks for imports, MuJoCo environment creation, data loading, and a
  minimal training pass. The directory is created only when the smoke script is
  run.

cheetah_dir_demo/
  Completed local demonstration runs for the `cheetah_dir` benchmark.
```
