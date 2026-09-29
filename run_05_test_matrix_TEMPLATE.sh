#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
V20_CKPT="/path/to/local/storage/SubsetSynth_v2_0b_runs/FULL_RUN/checkpoints/best_mapping_balanced_psnr.pth"
UNLOCK="YES_I_CONFIRM_MODEL_AND_INFERENCE_ARE_FROZEN"
python scripts/evaluate_v2.py --config configs/base.yaml --checkpoint "$V20_CKPT" --split test --matrix configs/evaluation_configs.yaml --with-lpips --unlock-test "$UNLOCK" --out /path/to/local/storage/v20b_test_matrix
