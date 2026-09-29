#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/preflight.py --config configs/profile.yaml
python scripts/run_profile.py --config configs/profile.yaml
