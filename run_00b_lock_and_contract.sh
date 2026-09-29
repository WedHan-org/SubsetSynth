#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
AUDIT=/path/to/local/storage/SubsetSynth_v2_0b_semantics_audit_v2_orientation/SEMANTICS_AUDIT_DRAFT.json
python scripts/approve_npy_semantics.py --config configs/base.yaml --audit-draft "$AUDIT" --confirm I_HAVE_VISUALLY_VERIFIED_NPY_MODALITY_SEMANTICS_AND_ORIENTATION
python scripts/build_support_lock.py --config configs/base.yaml
python scripts/preflight.py --config configs/base.yaml
python scripts/contract_tests.py --config configs/profile.yaml --device cpu --image-size 64
