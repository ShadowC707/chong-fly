#!/usr/bin/env bash
# ==============================================================================
# Chong-Fly: NVIDIA Isaac Gym Runner (RTX GPU PhysX Simulation)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/mmaksimp027/.venv-isaac/bin:$PATH"

# Run with Isaac Gym PhysX GPU pipeline enabled
exec /home/mmaksimp027/.venv-isaac/bin/python "${SCRIPT_DIR}/simulation/drone_env.py" --engine isaacgym "$@"
