#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash run_03_resume_recovery.sh /absolute/path/to/latest_resume.pth"
  exit 2
fi
RESUME="$1"
cd "$(dirname "$0")"
python scripts/preflight.py --config configs/train.yaml
python scripts/prepare_resume_after_amp_recovery.py --resume "$RESUME"
python scripts/train.py --config configs/train.yaml --resume "$RESUME"
