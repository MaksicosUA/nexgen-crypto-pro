#!/usr/bin/env bash
# Run a command with the CUDA library path set up correctly.
#
# Why this is needed: TensorFlow 2.21 loads CUDA libraries via dlopen and does
# not add every nvidia-* package directory to the search path on its own. In
# particular, it cannot find libcusolver.so.11, even though pip installs it into
#   .venv/lib/python3.12/site-packages/nvidia/cusolver/lib/
# A single missing library is enough for TF to silently give up on the GPU:
#   Cannot dlopen some GPU libraries ... Skipping registering GPU devices
#   GPU devices: []
# The other ten libraries still load successfully, so the log makes it look
# as if CUDA is fine.
#
# This script adds every lib directory from the nvidia-* packages to LD_LIBRARY_PATH.
#
# Usage:
#   scripts/with_gpu.sh .venv/bin/python models/train_gru.py --config ... --file ...
#   scripts/with_gpu.sh .venv/bin/python controller/auto_retrain.py --models gru
#
# Child processes inherit the environment, so auto_retrain.py, which runs
# training and the validation gate via subprocess, also gets the GPU.
#
# Check that the GPU is visible:
#   scripts/with_gpu.sh .venv/bin/python -c \
#     "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVIDIA_ROOT="$PROJECT_DIR/.venv/lib/python3.12/site-packages/nvidia"

if [[ ! -d "$NVIDIA_ROOT" ]]; then
  echo "CUDA packages not found: $NVIDIA_ROOT" >&2
  echo "Install with: .venv/bin/python -m pip install 'tensorflow[and-cuda]==2.21.0'" >&2
  exit 1
fi

CUDA_LIBS="$(find "$NVIDIA_ROOT" -type d -name lib -printf '%p:' 2>/dev/null)"

if [[ -z "$CUDA_LIBS" ]]; then
  echo "No lib directories found in $NVIDIA_ROOT" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:-}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi

exec "$@"
