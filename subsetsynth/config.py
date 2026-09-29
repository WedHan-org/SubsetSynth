from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in child.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base = load_config(base_path)
        cfg = _deep_merge(base, cfg)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    mods = cfg.get("model", {}).get("modalities", [])
    if mods != ["T1", "T2", "FS"]:
        raise ValueError(f"Frozen vocabulary must be ['T1','T2','FS']; got {mods}")
    tr = cfg.get("training", {})
    if tr.get("max_epochs", 0) < 1:
        raise ValueError("training.max_epochs must be >=1")
    if tr.get("effective_batch_size", 0) < tr.get("micro_batch_size", 1):
        raise ValueError("effective_batch_size must be >= micro_batch_size")
    if tr.get("effective_batch_size", 1) % tr.get("micro_batch_size", 1) != 0:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size")
    proposal = cfg.get("sampling", {}).get("cohort_proposal", {})
    if set(proposal) != {"A","B","C2","C3"}:
        raise ValueError("sampling.cohort_proposal must define exactly A/B/C2/C3")
    if abs(sum(float(v) for v in proposal.values()) - 1.0) > 1e-12:
        raise ValueError("sampling.cohort_proposal must sum to 1")
    if any(float(v) <= 0 for v in proposal.values()):
        raise ValueError("sampling.cohort_proposal values must all be >0")
    if cfg.get("physics", {}).get("enabled", False) and float(cfg.get("physics", {}).get("weight", 0)) <= 0:
        raise ValueError("physics.weight must be >0 when physics.enabled=true")
    schema = cfg.get("data_schema", {})
    if set(schema) != {"A","B","C2","C3"}:
        raise ValueError("data_schema must define exactly A/B/C2/C3")


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_resolved_config(cfg: Dict[str, Any], path: str | Path) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)
