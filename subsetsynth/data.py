from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .semantics import canonicalize_orientation


COHORTS = {
    "A": 0,
    "B": 1,
    "C2": 2,
    "C3": 3,
}
COHORT_NAMES = {v: k for k, v in COHORTS.items()}
AVAIL_BY_COHORT = {
    0: (1, 1, 1),
    1: (1, 1, 1),
    2: (1, 1, 0),
    3: (1, 0, 1),
}


@dataclass(frozen=True)
class Record:
    path: str
    case_id: str
    cohort_id: int
    split: str


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, p1, p99)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo > 1e-5:
        arr = (arr - lo) / (hi - lo)
    return arr.astype(np.float32)


def parse_case_id(filename: str) -> str:
    if "Case_" in filename:
        return "_".join(filename.split("_")[:2])
    return filename.rsplit("_slice", 1)[0]


def load_split_map(csv_path: str | Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    required = {"case_id", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain {required}; got {list(df.columns)}")
    case_ids = df["case_id"].astype(str).str.strip()
    splits = df["split"].astype(str).str.strip().str.lower()
    if case_ids.duplicated().any():
        dup = case_ids[case_ids.duplicated()].tolist()[:10]
        raise ValueError(f"Duplicate case_id in {csv_path}: {dup}")
    bad = sorted(set(splits) - {"train", "val", "test"})
    if bad:
        raise ValueError(f"Unexpected split labels in {csv_path}: {bad}")
    return dict(zip(case_ids, splits))


class MultiContrastNPYDataset(Dataset):
    """Loads the recovered v1.8.2 NPY representation into canonical [T1,T2,FS]."""

    def __init__(self, cfg: dict, split: str, strict_split: bool = True, allow_unlocked_data: bool = False):
        self.cfg = cfg
        if not allow_unlocked_data:
            from .locks import load_semantics_lock
            load_semantics_lock(cfg, allow_unlocked=False)
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError(self.split)
        self.img_size = int(cfg["data"]["img_size"])
        self.records: List[Record] = []
        self.strict_split = bool(strict_split)

        specs = [
            ("A", cfg["data"]["A"]["dir"], cfg["data"]["A"]["split_csv"]),
            ("B", cfg["data"]["B"]["dir"], cfg["data"]["B"]["split_csv"]),
            ("C2", cfg["data"]["C2"]["dir"], cfg["data"]["C2"]["split_csv"]),
            ("C3", cfg["data"]["C3"]["dir"], cfg["data"]["C3"]["split_csv"]),
        ]
        for name, directory, csv_path in specs:
            self._load_cohort(name, directory, csv_path)

        if not self.records:
            raise RuntimeError(f"No records found for split={self.split}")

        self.cohort_counts = {i: 0 for i in range(4)}
        for r in self.records:
            self.cohort_counts[r.cohort_id] += 1

    def _load_cohort(self, name: str, directory: str, csv_path: str) -> None:
        cohort_id = COHORTS[name]
        directory = Path(directory)
        csv_path = Path(csv_path)
        if not directory.exists():
            raise FileNotFoundError(directory)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        split_map = load_split_map(csv_path)
        unknown = []
        for f in sorted(directory.iterdir()):
            if f.suffix.lower() != ".npy":
                continue
            case_id = parse_case_id(f.name)
            split = split_map.get(case_id)
            if split is None:
                unknown.append(case_id)
                continue
            if split == self.split:
                self.records.append(Record(str(f), case_id, cohort_id, split))
        if unknown and self.strict_split:
            uniq = sorted(set(unknown))
            raise RuntimeError(
                f"{name}: {len(uniq)} case IDs have NPY files but are absent from split CSV. "
                f"Examples: {uniq[:10]}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _resize_norm(self, arr: np.ndarray) -> np.ndarray:
        resized = [cv2.resize(arr[c], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                   for c in range(arr.shape[0])]
        return np.stack([normalize_to_01(x) for x in resized], axis=0).astype(np.float32)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        loaded = np.load(r.path, allow_pickle=True).item()
        if "input" not in loaded or "label" not in loaded:
            raise KeyError(f"Expected input/label keys in {r.path}")
        inp, gt = loaded["input"], loaded["label"]

        if inp.ndim == 2:
            inp = inp[np.newaxis, ...]
        elif inp.ndim == 3 and inp.shape[-1] <= 3:
            inp = inp.transpose(2, 0, 1)
        if gt.ndim == 2:
            gt = gt[np.newaxis, ...]
        elif gt.ndim == 3 and gt.shape[-1] <= 3:
            gt = gt.transpose(2, 0, 1)

        if r.cohort_id in (0, 1):
            if not (inp.shape[0] == 2 and gt.shape[0] == 1):
                raise AssertionError(f"A/B shape bad: {r.path}, inp={inp.shape}, gt={gt.shape}")
        else:
            if not (inp.shape[0] == 1 and gt.shape[0] == 1):
                raise AssertionError(f"C shape bad: {r.path}, inp={inp.shape}, gt={gt.shape}")

        # Canonical orientation is a shared, frozen data-semantic contract.
        # Dataset A preserves the historical v1.8.2 transform exactly; B/C2/C3 are identity.
        inp = canonicalize_orientation(inp, r.cohort_id)
        gt = canonicalize_orientation(gt, r.cohort_id)

        inp = self._resize_norm(inp)
        gt = self._resize_norm(gt)

        zeros = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
        if r.cohort_id in (0, 1):
            full = np.concatenate([inp, gt], axis=0)  # [T1,T2,FS]
        elif r.cohort_id == 2:
            full = np.concatenate([inp, gt, zeros], axis=0)  # [T1,T2,0]
        elif r.cohort_id == 3:
            full = np.concatenate([inp, zeros, gt], axis=0)  # [T1,0,FS]
        else:
            raise AssertionError(r.cohort_id)

        avail = np.asarray(AVAIL_BY_COHORT[r.cohort_id], dtype=np.bool_)
        full = full * avail[:, None, None].astype(np.float32)
        return {
            "images": torch.from_numpy(full.copy()),
            "avail_mask": torch.from_numpy(avail.copy()),
            "cohort_id": torch.tensor(r.cohort_id, dtype=torch.long),
            "case_id": r.case_id,
            "path": r.path,
        }


def training_sampler_plan(dataset: MultiContrastNPYDataset, cfg: dict) -> dict:
    counts = dataset.cohort_counts
    proposal = {COHORTS[k]: float(v) for k, v in cfg["sampling"]["cohort_proposal"].items()}
    missing = [COHORT_NAMES[c] for c,p in proposal.items() if p > 0 and counts.get(c,0) <= 0]
    if missing:
        raise RuntimeError(f"Proposal assigns probability to empty training cohorts: {missing}")
    effective = int(cfg["training"]["effective_batch_size"])
    n_records = len(dataset.records)
    num_samples = (n_records // effective) * effective
    if num_samples <= 0:
        raise RuntimeError(f"Not enough slice records ({n_records}) for effective_batch_size={effective}")
    weights = [proposal[r.cohort_id] / max(counts[r.cohort_id], 1) for r in dataset.records]
    return {
        "weights": weights,
        "num_samples": num_samples,
        "n_records": n_records,
        "effective_batch_size": effective,
        "cohort_record_counts": {COHORT_NAMES[k]: int(v) for k,v in counts.items()},
        "cohort_proposal": {COHORT_NAMES[k]: float(v) for k,v in proposal.items()},
    }


def make_train_loader(dataset: MultiContrastNPYDataset, cfg: dict, epoch: int) -> DataLoader:
    plan = training_sampler_plan(dataset, cfg)
    gen = torch.Generator()
    gen.manual_seed(int(cfg["reproducibility"]["seed"]) + int(epoch) * 1009)
    sampler = WeightedRandomSampler(
        weights=plan["weights"],
        num_samples=plan["num_samples"],
        replacement=True,
        generator=gen,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["micro_batch_size"]),
        sampler=sampler,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", True)) and int(cfg["data"]["num_workers"]) > 0,
        drop_last=True,
    )
    accum = int(cfg["training"]["effective_batch_size"]) // int(cfg["training"]["micro_batch_size"])
    if len(loader) % accum != 0:
        raise AssertionError(
            f"Training loader has partial accumulation window: batches={len(loader)}, accum_steps={accum}"
        )
    return loader


def make_eval_loader(dataset: MultiContrastNPYDataset, cfg: dict) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg["evaluation"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", True)) and int(cfg["data"]["num_workers"]) > 0,
        drop_last=False,
    )


def paired_training_augment(images: torch.Tensor, avail_mask: torch.Tensor, src_mask: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Role-symmetric paired augmentation plus source-only Rician noise.

    Gamma/bias are shared across modalities in a case; Rician noise is applied only to observed source slots.
    Targets remain aligned and source/target roles are not hard-coded to stored input/label roles.
    """
    aug = cfg.get("augmentation", {})
    if not aug.get("enabled", True):
        return images
    x = images.clone()
    B, _, H, W = x.shape
    device = x.device
    p = float(aug.get("probability", 0.6))
    active = torch.rand(B, device=device) < p
    if not active.any():
        return x

    gamma_lo, gamma_hi = aug.get("gamma", [0.85, 1.15])
    gamma = torch.empty(B, 1, 1, 1, device=device).uniform_(float(gamma_lo), float(gamma_hi))
    xa = torch.pow(torch.clamp(x, 0, 1), gamma)

    if float(aug.get("bias_strength", 0.15)) > 0:
        yy = torch.linspace(-1, 1, H, device=device)
        xx = torch.linspace(-1, 1, W, device=device)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        ax = torch.empty(B, 1, 1, 1, device=device).uniform_(-1, 1)
        ay = torch.empty(B, 1, 1, 1, device=device).uniform_(-1, 1)
        strength = float(aug.get("bias_strength", 0.15))
        bias = 1.0 + strength * (ax * X.view(1, 1, H, W) + ay * Y.view(1, 1, H, W))
        xa = torch.clamp(xa * bias, 0, 1)

    active4 = active.view(B, 1, 1, 1)
    x = torch.where(active4, xa, x)
    x = x * avail_mask[:, :, None, None].float()

    sigma = float(aug.get("source_rician_sigma", 0.015))
    if sigma > 0:
        noise_r = torch.randn_like(x) * sigma
        noise_i = torch.randn_like(x) * sigma
        noisy = torch.sqrt(torch.clamp((x + noise_r) ** 2 + noise_i ** 2, min=0.0))
        source_pixels = src_mask[:, :, None, None] & active4.bool()
        x = torch.where(source_pixels, torch.clamp(noisy, 0, 1), x)
    return x
