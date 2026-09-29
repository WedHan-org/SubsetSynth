from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

MODALITIES = ("T1", "T2", "FS")
MOD_TO_IDX = {m: i for i, m in enumerate(MODALITIES)}


def mask_key(mask: Sequence[bool | int]) -> str:
    return "".join("1" if bool(x) else "0" for x in mask)


def tensor_mask_key(mask: torch.Tensor) -> str:
    return mask_key(mask.detach().cpu().bool().tolist())


def nonempty_proper_subsets(avail_mask: Sequence[bool | int]) -> List[Tuple[int, int, int]]:
    avail = [i for i, x in enumerate(avail_mask) if bool(x)]
    if len(avail) < 2:
        raise ValueError(f"Need at least 2 available modalities, got {avail_mask}")
    out: List[Tuple[int, int, int]] = []
    for r in range(1, len(avail)):
        for combo in itertools.combinations(avail, r):
            m = [0, 0, 0]
            for i in combo:
                m[i] = 1
            out.append(tuple(m))
    return out


def primitive_mappings_for_avail(avail_mask: Sequence[bool | int]):
    """Derive primitive mappings algorithmically; no task table is used."""
    avail = tuple(bool(x) for x in avail_mask)
    mappings = []
    for src in nonempty_proper_subsets(avail):
        for t in range(3):
            if avail[t] and not src[t]:
                mappings.append((src, t))
    return mappings


def mapping_name(src_mask: Sequence[bool | int], target_idx: int) -> str:
    src = "+".join(MODALITIES[i] for i, x in enumerate(src_mask) if bool(x))
    return f"{src}->{MODALITIES[target_idx]}"


@dataclass
class SampledMask:
    src_mask: Tuple[int, int, int]
    tgt_mask: Tuple[int, int, int]


class SubsetBagManager:
    """Randomized subset bags keyed by cohort + availability pattern.

    Each bag contains every admissible source subset exactly once, shuffled.
    This preserves local uniformity without introducing predefined source-target task IDs.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self._bags: Dict[Tuple[int, str], List[Tuple[int, int, int]]] = defaultdict(list)
        self._cursor: Dict[Tuple[int, str], int] = defaultdict(int)
        self.exposure = Counter()

    def _refresh(self, key: Tuple[int, str], avail: Tuple[int, int, int]) -> None:
        bag = nonempty_proper_subsets(avail)
        self.rng.shuffle(bag)
        self._bags[key] = bag
        self._cursor[key] = 0

    def sample_one(self, cohort_id: int, avail_mask: Sequence[bool | int]) -> SampledMask:
        avail = tuple(int(bool(x)) for x in avail_mask)
        key = (int(cohort_id), mask_key(avail))
        if key not in self._bags or self._cursor[key] >= len(self._bags[key]):
            self._refresh(key, avail)
        src = self._bags[key][self._cursor[key]]
        self._cursor[key] += 1
        tgt = tuple(int(a and not s) for a, s in zip(avail, src))
        for t, active in enumerate(tgt):
            if active:
                self.exposure[mapping_name(src, t)] += 1
        return SampledMask(src, tgt)

    def sample_batch(self, cohort_ids: torch.Tensor, avail_masks: torch.Tensor):
        srcs, tgts = [], []
        for c, a in zip(cohort_ids.detach().cpu().tolist(), avail_masks.detach().cpu().tolist()):
            item = self.sample_one(int(c), a)
            srcs.append(item.src_mask)
            tgts.append(item.tgt_mask)
        device = avail_masks.device
        return (
            torch.tensor(srcs, dtype=torch.bool, device=device),
            torch.tensor(tgts, dtype=torch.bool, device=device),
        )

    def state_dict(self):
        return {
            "seed": self.seed,
            "rng_state": self.rng.getstate(),
            "bags": dict(self._bags),
            "cursor": dict(self._cursor),
            "exposure": dict(self.exposure),
        }

    def load_state_dict(self, state):
        self.seed = int(state["seed"])
        self.rng = random.Random()
        self.rng.setstate(state["rng_state"])
        self._bags = defaultdict(list, state.get("bags", {}))
        self._cursor = defaultdict(int, state.get("cursor", {}))
        self.exposure = Counter(state.get("exposure", {}))
