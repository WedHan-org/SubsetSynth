from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import pandas as pd
import torch
from tqdm import tqdm

from .data import COHORT_NAMES, AVAIL_BY_COHORT
from .masks import nonempty_proper_subsets, mapping_name
from .metrics import MetricComputer, case_aggregate, mapping_cohort_balanced_summary
from .utils import dump_json


@torch.no_grad()
def evaluate_primitives(
    model,
    loader,
    cfg: dict,
    device,
    with_lpips: bool = False,
    desc: str = "eval",
    support_lock: Mapping | None = None,
):
    model.eval()
    metric = MetricComputer(cfg, with_lpips=with_lpips, device=device)
    slice_rows: List[dict] = []

    for batch in tqdm(loader, desc=desc):
        images = batch["images"].to(device, non_blocking=True).float()
        cohort_ids = batch["cohort_id"].to(device)
        for cohort in sorted(set(cohort_ids.detach().cpu().tolist())):
            idx = torch.where(cohort_ids == cohort)[0]
            if idx.numel() == 0:
                continue
            x = images[idx]
            avail = AVAIL_BY_COHORT[int(cohort)]
            for src_tuple in nonempty_proper_subsets(avail):
                src = torch.tensor(src_tuple, dtype=torch.bool, device=device).view(1,3).expand(idx.numel(),-1)
                z, skips = model.encode(x, src)
                for target_idx in range(3):
                    if not avail[target_idx] or src_tuple[target_idx]:
                        continue
                    t = torch.full((idx.numel(),), int(target_idx), device=device, dtype=torch.long)
                    pred = model.decode(z, skips, t)
                    gt = x[:,target_idx:target_idx+1]
                    vals = metric.compute(pred, gt)
                    for local_j, original_j in enumerate(idx.detach().cpu().tolist()):
                        row = {
                            "case_id": batch["case_id"][original_j],
                            "cohort": COHORT_NAMES[int(cohort)],
                            "mapping": mapping_name(src_tuple, target_idx),
                            "psnr": float(vals["psnr"][local_j].cpu()),
                            "ssim": float(vals["ssim"][local_j].cpu()),
                            "mae": float(vals["mae"][local_j].cpu()),
                        }
                        if "lpips" in vals:
                            row["lpips"] = float(vals["lpips"][local_j].cpu())
                        slice_rows.append(row)
    cases = case_aggregate(slice_rows)
    summary = mapping_cohort_balanced_summary(cases, support_lock=support_lock, strict=True)
    return slice_rows, cases, summary


def save_evaluation(rows, cases, summary, out_dir: str | Path, prefix: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_slice_metrics.csv", index=False)
    pd.DataFrame(cases).to_csv(out_dir / f"{prefix}_case_metrics.csv", index=False)
    dump_json(summary, out_dir / f"{prefix}_summary.json")
