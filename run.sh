#!/usr/bin/env bash
# Run a command in the upstream microduck_rl venv, from this repo's root (Linux / WSL).
#   bash run.sh python scripts/walk_carry.py beak
# The venv is the one `uv sync` made for microduck_rl (MICRODUCK_VENV, default
# ~/.venvs/microduck_rl). Its python is used directly, so nothing gets synced or installed.
export PATH="${MICRODUCK_VENV:-$HOME/.venvs/microduck_rl}/bin:$HOME/.local/bin:$PATH"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$(dirname "$0")"
exec "$@"
