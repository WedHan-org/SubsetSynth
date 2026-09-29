from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

from .masks import mapping_name, primitive_mappings_for_avail
from .utils import sha256_file
from .semantics import orientation_contract


SEMANTIC_LOCK_VERSION = "SubsetSynthNPYSemanticsLock_v2_orientation"
SUPPORT_LOCK_VERSION = "SubsetSynthMappingCohortSupportLock_v1"
COHORTS = {"A":0,"B":1,"C2":2,"C3":3}
AVAIL_BY_COHORT = {0:(1,1,1),1:(1,1,1),2:(1,1,0),3:(1,0,1)}


def _read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_semantics_lock(cfg: Mapping, allow_unlocked: bool = False) -> dict:
    p = Path(cfg["locks"]["npy_semantics_lock"])
    if allow_unlocked:
        return {"version": SEMANTIC_LOCK_VERSION, "approved": True, "synthetic_bypass": True}
    lock = _read_json(p)
    if lock.get("version") != SEMANTIC_LOCK_VERSION or not bool(lock.get("approved", False)):
        raise RuntimeError(f"NPY semantic lock is missing/unapproved/unsupported: {p}")
    expected = cfg["data_schema"]
    if lock.get("schema") != expected:
        raise RuntimeError("NPY semantic lock schema does not match config data_schema")
    expected_orientation = orientation_contract()
    if lock.get("orientation_contract") != expected_orientation:
        raise RuntimeError("NPY semantic lock orientation contract does not match shared model/audit code")
    # A split hash mismatch means the audit/approval provenance is stale.
    locked_hashes = lock.get("split_sha256", {})
    for cohort in COHORTS:
        split_path = Path(cfg["data"][cohort]["split_csv"])
        current = sha256_file(split_path)
        if locked_hashes.get(cohort) != current:
            raise RuntimeError(f"NPY semantic lock stale for {cohort}: split hash changed")
    return lock


def expected_mapping_support() -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    for cohort, cid in COHORTS.items():
        for src, target in primitive_mappings_for_avail(AVAIL_BY_COHORT[cid]):
            out.setdefault(mapping_name(src, target), []).append(cohort)
    return {m: sorted(cs) for m, cs in sorted(out.items())}


def load_support_lock(cfg: Mapping, allow_unlocked: bool = False) -> dict:
    p = Path(cfg["locks"]["mapping_cohort_support_lock"])
    if allow_unlocked:
        support = expected_mapping_support()
        return {
            "version": SUPPORT_LOCK_VERSION,
            "approved": True,
            "synthetic_bypass": True,
            "mapping_order": sorted(support),
            "mapping_support": support,
        }
    lock = _read_json(p)
    if lock.get("version") != SUPPORT_LOCK_VERSION or not bool(lock.get("approved", False)):
        raise RuntimeError(f"Mapping-cohort support lock is missing/unapproved/unsupported: {p}")
    expected = expected_mapping_support()
    got = {m: sorted(cs) for m, cs in lock.get("mapping_support", {}).items()}
    if got != expected:
        raise RuntimeError(
            "Frozen mapping-cohort support differs from the theoretical availability contract. "
            f"expected={expected}, got={got}"
        )
    if set(lock.get("mapping_order", [])) != set(expected):
        raise RuntimeError("Support lock mapping_order does not contain exactly the nine primitive mappings")
    locked_split = lock.get("split_sha256", {})
    for cohort in COHORTS:
        current = sha256_file(cfg["data"][cohort]["split_csv"])
        if locked_split.get(cohort) != current:
            raise RuntimeError(f"Mapping support lock stale for {cohort}: split hash changed")
    sem_path = Path(cfg["locks"]["npy_semantics_lock"])
    if lock.get("npy_semantics_lock_sha256") != sha256_file(sem_path):
        raise RuntimeError("Mapping support lock stale: NPY semantic lock changed")
    # Validation must contain at least one materialized case for every frozen cell.
    val_counts = lock.get("validation_case_counts_by_mapping_cohort", {})
    missing = []
    for m, cohorts in expected.items():
        for c in cohorts:
            if int(val_counts.get(f"{m}|{c}", 0)) <= 0:
                missing.append(f"{m}|{c}")
    if missing:
        raise RuntimeError(f"Support lock has empty validation cells: {missing}")
    return lock
