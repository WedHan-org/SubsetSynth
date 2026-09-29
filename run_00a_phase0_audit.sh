#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/preflight.py --config configs/base.yaml --allow-missing-locks
python scripts/audit_splits.py --config configs/base.yaml --out /path/to/local/storage/SubsetSynth_v2_0b_phase0_audit
python scripts/audit_npy_semantics.py --config configs/base.yaml --out /path/to/local/storage/SubsetSynth_v2_0b_semantics_audit_v2_orientation --samples-per-cohort 6
cat <<'EOF'
STOP HERE. Visually review every CANONICAL PNG in:
/path/to/local/storage/SubsetSynth_v2_0b_semantics_audit_v2_orientation/contact_sheets
Raw-NPY orientation is retained separately in:
/path/to/local/storage/SubsetSynth_v2_0b_semantics_audit_v2_orientation/raw_contact_sheets
Confirm both modality identity AND canonical orientation before running run_00b_lock_and_contract.sh.
EOF
