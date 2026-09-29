from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping

import numpy as np
import torch

from .losses import DifferentiableSSIM


class MetricComputer:
    def __init__(self, cfg: dict, with_lpips: bool = False, device=None):
        self.ssim = DifferentiableSSIM(
            window_size=int(cfg["evaluation"]["ssim_window"]),
            sigma=float(cfg["evaluation"]["ssim_sigma"]),
        ).to(device or "cpu")
        self.with_lpips = with_lpips
        self.lpips_model = None
        if with_lpips:
            try:
                import lpips
            except ImportError as e:
                raise RuntimeError("LPIPS requested but package 'lpips' is not installed. pip install lpips") from e
            net = cfg["evaluation"].get("lpips_net", "alex")
            self.lpips_model = lpips.LPIPS(net=net).to(device or "cpu").eval()

    @torch.no_grad()
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse = (pred-target).square().flatten(1).mean(1).clamp(min=1e-12)
        psnr = 10.0 * torch.log10(1.0/mse)
        ssim = self.ssim.per_image(pred,target)
        mae = (pred-target).abs().flatten(1).mean(1)
        out = {"psnr": psnr, "ssim": ssim, "mae": mae}
        if self.lpips_model is not None:
            p = pred.repeat(1,3,1,1)*2-1
            t = target.repeat(1,3,1,1)*2-1
            out["lpips"] = self.lpips_model(p,t).view(-1)
        return out


def case_aggregate(rows: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["case_id"], r["cohort"], r["mapping"])].append(r)
    out = []
    for (case_id, cohort, mapping), rs in groups.items():
        d = {"case_id": case_id, "cohort": cohort, "mapping": mapping, "n_slices": len(rs)}
        for k in ("psnr","ssim","mae","lpips"):
            vals = [x[k] for x in rs if k in x and x[k] is not None and np.isfinite(x[k])]
            if vals:
                d[k] = float(np.mean(vals))
        out.append(d)
    return out


def validate_case_metric_rows(case_rows: List[dict], require_lpips: bool = False) -> None:
    if not case_rows:
        raise ValueError("Empty case metric table")
    required = {"case_id","cohort","mapping","psnr","ssim","mae"}
    if require_lpips:
        required.add("lpips")
    seen = set()
    for i, r in enumerate(case_rows):
        missing = required - set(r)
        if missing:
            raise ValueError(f"Case metric row {i} missing columns/values: {sorted(missing)}")
        key = (str(r["case_id"]), str(r["cohort"]), str(r["mapping"]))
        if key in seen:
            raise ValueError(f"Duplicate case/cohort/mapping row: {key}")
        seen.add(key)
        for metric in ("psnr","ssim","mae") + (("lpips",) if require_lpips else ()): 
            if not np.isfinite(float(r[metric])):
                raise ValueError(f"Non-finite {metric} for {key}")


def mapping_cohort_balanced_summary(case_rows: List[dict], support_lock: Mapping | None = None, strict: bool = True) -> dict:
    """Case -> mapping×cohort -> cohort-balanced mapping -> mapping-balanced overall.

    With a support lock, the aggregation set is frozen. Empty expected cells fail closed;
    unexpected cells also fail in strict mode. Thus |C_m| cannot silently change.
    """
    by_mc = defaultdict(list)
    for r in case_rows:
        by_mc[(str(r["mapping"]), str(r["cohort"]))].append(r)

    expected_cells = None
    mapping_order = None
    if support_lock is not None:
        support = {str(m): tuple(str(c) for c in cs) for m,cs in support_lock["mapping_support"].items()}
        mapping_order = list(support_lock.get("mapping_order", support.keys()))
        expected_cells = {(m,c) for m,cs in support.items() for c in cs}
        got_cells = set(by_mc)
        missing = sorted(expected_cells - got_cells)
        extra = sorted(got_cells - expected_cells)
        if missing:
            raise RuntimeError(f"Validation/evaluation missing frozen mapping×cohort cells: {missing}")
        if strict and extra:
            raise RuntimeError(f"Unexpected mapping×cohort cells outside frozen support: {extra}")

    mc = {}
    for key, rs in by_mc.items():
        if expected_cells is not None and key not in expected_cells:
            continue
        mc[key] = {
            k: float(np.mean([float(r[k]) for r in rs if k in r and r[k] is not None and np.isfinite(float(r[k]))]))
            for k in ("psnr","ssim","mae","lpips")
            if any(k in r and r[k] is not None and np.isfinite(float(r[k])) for r in rs)
        }

    if support_lock is None:
        support = defaultdict(list)
        for (m,c) in mc:
            support[m].append(c)
        support = {m: tuple(sorted(cs)) for m,cs in support.items()}
        mapping_order = sorted(support)
    else:
        support = {str(m): tuple(str(c) for c in cs) for m,cs in support_lock["mapping_support"].items()}

    mapping = {}
    for m in mapping_order:
        vals = [mc[(m,c)] for c in support[m]]
        mapping[m] = {
            k: float(np.mean([v[k] for v in vals if k in v]))
            for k in ("psnr","ssim","mae","lpips")
            if any(k in v for v in vals)
        }

    overall = {
        k: float(np.mean([mapping[m][k] for m in mapping_order if k in mapping[m]]))
        for k in ("psnr","ssim","mae","lpips")
        if any(k in mapping[m] for m in mapping_order)
    }

    by_cohort = {}
    cohorts = sorted({c for cs in support.values() for c in cs})
    for c in cohorts:
        maps = [m for m in mapping_order if c in support[m]]
        by_cohort[c] = {
            k: float(np.mean([mc[(m,c)][k] for m in maps if k in mc[(m,c)]]))
            for k in ("psnr","ssim","mae","lpips")
            if any(k in mc[(m,c)] for m in maps)
        }
        by_cohort[c]["n_supported_mappings"] = len(maps)

    return {
        "overall_mapping_balanced": overall,
        "mapping": mapping,
        "mapping_cohort": {f"{m}|{c}":v for (m,c),v in mc.items()},
        "cohort_mapping_balanced": by_cohort,
    }
