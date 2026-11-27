#!/usr/bin/env bash
# Eye Gaze Attention Monitor launcher (Linux / macOS).
# First run creates .venv and installs dependencies. Arguments pass through, e.g.
#   ./run.sh --record
#   ./run.sh --source samples/desk-worker.mp4 --calibrate
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    if command -v nvidia-smi >/dev/null 2>&1; then
        .venv/bin/python -m pip install -r requirements.txt
    else
        .venv/bin/python -m pip install -r requirements-cpu.txt
    fi
fi

PYTHONPATH="$(pwd)/src" exec .venv/bin/python -m gaze_monitor "$@"
