#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/preflight.py --config configs/pilot.yaml
python scripts/train.py --config configs/pilot.yaml
