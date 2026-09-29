from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .data import COHORTS, COHORT_NAMES, AVAIL_BY_COHORT
from .masks import mapping_name, nonempty_proper_subsets, primitive_mappings_for_avail


@dataclass(frozen=True)
class ObjectiveCell:
    mapping: str
    cohort: str
    q: float       # proposal probability that this active target cell appears per sampled record
    alpha: float   # desired scientific target mass
    weight: float  # alpha / q


class HierarchicalImportanceObjective:
    """Fixed-target hierarchical objective used by v2.0b.

    Scientific target:
      1) all nine primitive mappings are equally weighted;
      2) within a mapping, every frozen supporting cohort is equally weighted.

    The cohort sampler + uniform source-subset sampler define only a proposal q(m,c).
    Importance weights a(m,c)=alpha(m,c)/q(m,c) convert that proposal into the
    frozen scientific target. No task ID is passed into the model.
    """

    def __init__(self, cohort_probabilities: Mapping[str, float], support_lock: Mapping):
        self.cohort_probabilities = {str(k): float(v) for k, v in cohort_probabilities.items()}
        expected_cohorts = set(COHORTS)
        if set(self.cohort_probabilities) != expected_cohorts:
            raise ValueError(
                f"cohort proposal must define exactly {sorted(expected_cohorts)}; "
                f"got {sorted(self.cohort_probabilities)}"
            )
        if any(v <= 0 for v in self.cohort_probabilities.values()):
            raise ValueError("all cohort proposal probabilities must be >0")
        total_p = sum(self.cohort_probabilities.values())
        if abs(total_p - 1.0) > 1e-12:
            raise ValueError(f"cohort proposal probabilities must sum to 1; got {total_p}")

        if not bool(support_lock.get("approved", False)):
            raise ValueError("mapping-cohort support lock is not approved")
        self.mapping_support = {
            str(m): tuple(str(c) for c in cohorts)
            for m, cohorts in support_lock["mapping_support"].items()
        }
        self.mapping_order = tuple(support_lock.get("mapping_order", self.mapping_support.keys()))
        if len(self.mapping_order) != 9 or set(self.mapping_order) != set(self.mapping_support):
            raise ValueError("Frozen v2.0b objective requires exactly 9 primitive mappings")

        self._num_subsets = {}
        for c_name, c_id in COHORTS.items():
            self._num_subsets[c_name] = len(nonempty_proper_subsets(AVAIL_BY_COHORT[c_id]))

        cells: Dict[Tuple[str, str], ObjectiveCell] = {}
        n_mappings = len(self.mapping_order)
        for m in self.mapping_order:
            supporters = self.mapping_support[m]
            if not supporters:
                raise ValueError(f"mapping {m} has no supporting cohort in lock")
            alpha = 1.0 / (n_mappings * len(supporters))
            for cohort in supporters:
                if cohort not in COHORTS:
                    raise ValueError(f"Unknown cohort {cohort} in support lock")
                # For a fixed primitive mapping within one availability pattern, the mapping
                # appears whenever its source subset is drawn. All admissible source subsets
                # are uniform within cohort.
                q = self.cohort_probabilities[cohort] / self._num_subsets[cohort]
                cells[(m, cohort)] = ObjectiveCell(m, cohort, q, alpha, alpha / q)
        self.cells = cells

        expected_mass = sum(cell.q * cell.weight for cell in cells.values())
        if abs(expected_mass - 1.0) > 1e-12:
            raise AssertionError(f"importance normalization identity failed: E_q[a]={expected_mass}")
        self.expected_mass_per_sample = expected_mass

    @classmethod
    def from_config(cls, cfg: Mapping, support_lock: Mapping):
        proposal = cfg["sampling"]["cohort_proposal"]
        return cls(proposal, support_lock)

    def cell(self, mapping: str, cohort: str) -> ObjectiveCell:
        key = (str(mapping), str(cohort))
        if key not in self.cells:
            raise KeyError(f"Unsupported/frozen-invalid objective cell: {key}")
        return self.cells[key]

    def weight_for(self, src_mask: Sequence[int | bool], target_idx: int, cohort_id: int) -> float:
        m = mapping_name(src_mask, int(target_idx))
        c = COHORT_NAMES[int(cohort_id)]
        return self.cell(m, c).weight

    def q_for(self, mapping: str, cohort: str) -> float:
        return self.cell(mapping, cohort).q

    def alpha_for(self, mapping: str, cohort: str) -> float:
        return self.cell(mapping, cohort).alpha

    def mapping_expected_raw_exposure(self) -> Dict[str, float]:
        out = {m: 0.0 for m in self.mapping_order}
        for (m, _), cell in self.cells.items():
            out[m] += cell.q
        return out

    def mapping_expected_target_mass(self) -> Dict[str, float]:
        out = {m: 0.0 for m in self.mapping_order}
        for (m, _), cell in self.cells.items():
            out[m] += cell.alpha
        return out

    def cohort_expected_target_mass(self) -> Dict[str, float]:
        out = {c: 0.0 for c in COHORTS}
        for (_, c), cell in self.cells.items():
            out[c] += cell.alpha
        return out

    def to_dict(self) -> dict:
        return {
            "definition": "alpha(m,c)=1/(9*|C_m|); q(m,c)=P(cohort=c)/N_source_subsets(c); a=alpha/q",
            "cohort_proposal": self.cohort_probabilities,
            "expected_mass_per_sample": self.expected_mass_per_sample,
            "mapping_expected_raw_exposure": self.mapping_expected_raw_exposure(),
            "mapping_expected_target_mass": self.mapping_expected_target_mass(),
            "cohort_expected_target_mass": self.cohort_expected_target_mass(),
            "cells": [
                {
                    "mapping": cell.mapping,
                    "cohort": cell.cohort,
                    "q": cell.q,
                    "alpha": cell.alpha,
                    "weight": cell.weight,
                }
                for cell in sorted(self.cells.values(), key=lambda x: (x.mapping, x.cohort))
            ],
        }


def theoretical_support_from_availability() -> Dict[str, Tuple[str, ...]]:
    support: Dict[str, list[str]] = {}
    for c_name, c_id in COHORTS.items():
        for src, t in primitive_mappings_for_avail(AVAIL_BY_COHORT[c_id]):
            support.setdefault(mapping_name(src, t), []).append(c_name)
    return {m: tuple(cs) for m, cs in support.items()}
