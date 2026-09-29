from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


# Frozen cohort-specific orientation contract.
# IMPORTANT: Dataset A reproduces the historical v1.8.2 model-input convention exactly:
#   1) flip CHW axis=1
#   2) rotate 90 degrees clockwise in spatial axes (1,2)
# B/C2/C3 are already stored in the canonical orientation used by the model.
_ORIENTATION_CONTRACT = {
    "A": {
        "stored_orientation": "legacy_A_npy_orientation",
        "canonicalization": ["flip_axis_1", "rot90_clockwise"],
    },
    "B": {
        "stored_orientation": "canonical_model_orientation",
        "canonicalization": [],
    },
    "C2": {
        "stored_orientation": "canonical_model_orientation",
        "canonicalization": [],
    },
    "C3": {
        "stored_orientation": "canonical_model_orientation",
        "canonicalization": [],
    },
}

_COHORT_ID_TO_NAME = {0: "A", 1: "B", 2: "C2", 3: "C3"}


def orientation_contract() -> dict[str, dict[str, Any]]:
    """Return a copy so callers cannot mutate the frozen in-code contract."""
    return deepcopy(_ORIENTATION_CONTRACT)


def cohort_name(cohort: int | str) -> str:
    if isinstance(cohort, str):
        if cohort not in _ORIENTATION_CONTRACT:
            raise KeyError(f"Unknown cohort: {cohort}")
        return cohort
    try:
        return _COHORT_ID_TO_NAME[int(cohort)]
    except Exception as e:
        raise KeyError(f"Unknown cohort id: {cohort}") from e


def canonicalize_orientation(arr: np.ndarray, cohort: int | str) -> np.ndarray:
    """Convert a CHW array from cohort-native NPY orientation to model-canonical orientation.

    This is deliberately the single shared implementation used by both the training/evaluation
    loader and the Phase-0 semantic audit. It does NOT alter modality order or intensity values.
    """
    x = np.asarray(arr)
    if x.ndim != 3:
        raise ValueError(f"canonicalize_orientation expects CHW 3D array; got {x.shape}")

    name = cohort_name(cohort)
    if name == "A":
        # Historical v1.8.2 convention; preserve operation order exactly.
        x = np.flip(x, axis=1)
        x = np.rot90(x, k=-1, axes=(1, 2))

    return np.ascontiguousarray(x)
