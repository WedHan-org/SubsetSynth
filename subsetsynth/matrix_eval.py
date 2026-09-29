from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch
import yaml

from .data import COHORT_NAMES
from .masks import MOD_TO_IDX, mapping_name
from .metrics import MetricComputer
from collections import defaultdict
import numpy as np
from .utils import dump_json


def load_matrix(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mask(names):
    m = [0,0,0]
    for n in names:
        m[MOD_TO_IDX[n]] = 1
    return tuple(m)


@torch.no_grad()
def evaluate_external_matrix(model, loader, cfg, device, matrix_path, with_lpips=True):
    matrix = load_matrix(matrix_path)
    configs = []
    for group in ("primitive_single_target", "compositional_multi_target"):
        for i, item in enumerate(matrix.get(group, [])):
            configs.append((group, i+1, item["source"], item["target"]))
    metric = MetricComputer(cfg, with_lpips=with_lpips, device=device)
    rows = []
    for batch in loader:
        images = batch["images"].to(device).float()
        avail = batch["avail_mask"].to(device).bool()
        cohort = batch["cohort_id"].to(device)
        for group, number, src_names, target_names in configs:
            sm = _mask(src_names)
            needed = torch.tensor([int(x) for x in sm], device=device, dtype=torch.bool)
            target_indices = [MOD_TO_IDX[t] for t in target_names]
            eligible = (avail | ~needed).all(dim=1)
            for t in target_indices:
                eligible = eligible & avail[:,t] & (~needed[t])
            idx = torch.where(eligible)[0]
            if idx.numel() == 0:
                continue
            src = needed.view(1,3).expand(idx.numel(),-1)
            x = images[idx]
            z, skips = model.encode(x,src)
            for t in target_indices:
                tid = torch.full((idx.numel(),),t,device=device,dtype=torch.long)
                pred = model.decode(z,skips,tid)
                gt = x[:,t:t+1]
                vals = metric.compute(pred,gt)
                for local_j, original_j in enumerate(idx.detach().cpu().tolist()):
                    rows.append({
                        "case_id": batch["case_id"][original_j],
                        "cohort": COHORT_NAMES[int(cohort[original_j].item())],
                        "configuration_group": group,
                        "configuration_id": number,
                        "source": "+".join(src_names),
                        "requested_targets": "+".join(target_names),
                        "mapping": mapping_name(sm,t),
                        "psnr": float(vals["psnr"][local_j].cpu()),
                        "ssim": float(vals["ssim"][local_j].cpu()),
                        "mae": float(vals["mae"][local_j].cpu()),
                        "lpips": float(vals["lpips"][local_j].cpu()) if "lpips" in vals else None,
                    })
    # Preserve configuration identity; the same primitive mapping can appear both alone and inside a compositional request.
    groups = defaultdict(list)
    for r in rows:
        key=(r["case_id"],r["cohort"],r["configuration_group"],r["configuration_id"],r["source"],r["requested_targets"],r["mapping"])
        groups[key].append(r)
    cases=[]
    for key,rs in groups.items():
        case_id,cohort,group,cid,source,requested,mapping=key
        d={"case_id":case_id,"cohort":cohort,"configuration_group":group,"configuration_id":cid,"source":source,"requested_targets":requested,"mapping":mapping,"n_slices":len(rs)}
        for metric_name in ("psnr","ssim","mae","lpips"):
            vals=[r[metric_name] for r in rs if r.get(metric_name) is not None and np.isfinite(r.get(metric_name))]
            if vals: d[metric_name]=float(np.mean(vals))
        cases.append(d)
    return rows, cases
