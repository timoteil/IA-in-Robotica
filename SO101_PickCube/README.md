# SO-101 pick-cube on Windows

Train the state-based `SO101PickCube` task (5-DoF SO-101 arm, pick up a cube) with MuJoCo Playground and PPO.

Do not copy `.venv` or `logs/wp_kernel_cache` from macOS. Create the environment on the Windows machine. `train.sh` is for macOS; the PowerShell commands below are the Windows equivalent.

## Prerequisites

1. [Git for Windows](https://git-scm.com/download/win). The first training run clones the SO-101 model from MuJoCo Menagerie.
2. [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **Desktop development with C++** workload. MuJoCo Warp compiles kernels on the first run.
3. [uv](https://docs.astral.sh/uv/getting-started/installation/). In PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal so `uv` is on `PATH`.

## Install

From this folder:

```powershell
cd SO101_PickCube
uv venv --python 3.13 .venv
.\.venv\Scripts\activate
uv pip install --python .\.venv\Scripts\python --default-index https://pypi.org/simple "playground[rl]>=1.0"
```

Playground needs Python 3.12 or 3.13. `uv` downloads 3.13 for the virtualenv.

## Train on CPU

Use this on a machine without an NVIDIA GPU. It runs 64 parallel arms. The GPU default is 8192 and will exhaust RAM.

```powershell
mkdir logs -Force
$env:PYTHONUNBUFFERED = "1"
python -m mujoco_playground.rl.rsl_rl.train `
  --env=SO101PickCube `
  --device=cpu `
  --logdir="$pwd\logs\so101_pick_cube" `
  --wp_kernel_cache_dir="$pwd\logs\wp_kernel_cache" `
  --overrides='{"nworld": 64}'
```

The first launch compiles Warp kernels and can sit quietly for a few minutes before iteration 0. Checkpoints land in `logs\so101_pick_cube` every 100 iterations (`model_100.pt`, …). Training is 1000 iterations.

Watch an existing log with:

```powershell
Get-Content .\logs\train.log -Wait
```

## Train on an NVIDIA GPU

Check the driver with `nvidia-smi`. The top of that output shows the CUDA version.

**CUDA 13.** The default install above is enough. Drop `--device` and the `nworld` override so it uses 8192 parallel environments:

```powershell
python -m mujoco_playground.rl.rsl_rl.train `
  --env=SO101PickCube `
  --logdir="$pwd\logs\so101_pick_cube" `
  --wp_kernel_cache_dir="$pwd\logs\wp_kernel_cache"
```

**CUDA 12.** Reinstall with the CUDA 12 PyTorch and Warp wheels, then use the same GPU command:

```powershell
uv pip install --python .\.venv\Scripts\python `
  --default-index https://pypi.org/simple `
  --index https://download.pytorch.org/whl/cu129 `
  -f https://github.com/NVIDIA/warp/releases/expanded_assets/v1.18.0 `
  "warp-lang==1.18.0+cu12" "playground[rl]>=1.0"
```

## Play a checkpoint

```powershell
python -m mujoco_playground.rl.rsl_rl.play `
  --env=SO101PickCube `
  --device=cpu `
  --resume_path=logs\so101_pick_cube
```

Omit `--device=cpu` on an NVIDIA GPU. This opens the policy in MuJoCo Studio in the browser.

Resume training from a checkpoint:

```powershell
python -m mujoco_playground.rl.rsl_rl.train `
  --env=SO101PickCube `
  --device=cpu `
  --logdir="$pwd\logs\so101_pick_cube" `
  --wp_kernel_cache_dir="$pwd\logs\wp_kernel_cache" `
  --overrides='{"nworld": 64}' `
  --resume_path=logs\so101_pick_cube\model_900.pt
```
