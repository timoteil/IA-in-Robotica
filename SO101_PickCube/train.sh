#!/bin/zsh
# State-based SO-101 pick-cube PPO. CPU path: 64 parallel worlds.
# GPU machines can drop --device and the nworld override (default is 8192).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONUNBUFFERED=1
exec .venv/bin/python -m mujoco_playground.rl.rsl_rl.train \
  --env=SO101PickCube \
  --device=cpu \
  --logdir="$(pwd)/logs/so101_pick_cube" \
  --wp_kernel_cache_dir="$(pwd)/logs/wp_kernel_cache" \
  --overrides='{"nworld": 64}' \
  "$@"
