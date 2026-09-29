#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "v2.0b requires a human NPY-semantic review between Phase 0a and 0b."
echo "Run: bash run_00a_phase0_audit.sh"
echo "After visually confirming contact sheets, run: bash run_00b_lock_and_contract.sh"
exit 2
