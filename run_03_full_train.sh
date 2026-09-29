#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "WARNING: full scientific run. Confirm profile/pilot review and freeze configs/train.yaml first."
python scripts/preflight.py --config configs/train.yaml
python scripts/train.py --config configs/train.yaml
