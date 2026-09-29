#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
LEGACY_SCRIPT="/path/to/local/storage/Any2Any_L_v1.8.2_Ultimate_cyecleGAN.py"
V18_CKPT="/path/to/local/storage/Transformer_MultiCenter_v1.8.2_Decoupled/LOCKED_CHECKPOINT.pth"
V20_CKPT="/path/to/local/storage/SubsetSynth_v2_0b_runs/FULL_RUN/checkpoints/best_mapping_balanced_psnr.pth"
python scripts/reevaluate_v18.py --config configs/base.yaml --legacy-script "$LEGACY_SCRIPT" --checkpoint "$V18_CKPT" --split val --with-lpips --out /path/to/local/storage/v18_common_eval_val
python scripts/evaluate_v2.py --config configs/base.yaml --checkpoint "$V20_CKPT" --split val --with-lpips --out /path/to/local/storage/v20b_common_eval_val
python scripts/gate_compare.py --config configs/base.yaml --v18-cases /path/to/local/storage/v18_common_eval_val/v18_val_primitive_case_metrics.csv --v20-cases /path/to/local/storage/v20b_common_eval_val/v2_val_primitive_case_metrics.csv --out /path/to/local/storage/v20b_gate_report.json
