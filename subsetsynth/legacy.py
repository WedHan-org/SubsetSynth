from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from .data import COHORT_NAMES, AVAIL_BY_COHORT
from .masks import primitive_mappings_for_avail, mapping_name
from .metrics import MetricComputer, case_aggregate, mapping_cohort_balanced_summary


LEGACY_TASK_BY_COHORT_AND_SRC = {
    0: {(1,0,0):0, (0,1,0):1, (0,0,1):2, (1,1,0):3, (1,0,1):4, (0,1,1):5},
    1: {(1,0,0):0, (0,1,0):1, (0,0,1):2, (1,1,0):3, (1,0,1):4, (0,1,1):5},
    2: {(1,0,0):6, (0,1,0):7},
    3: {(1,0,0):8, (0,0,1):9},
}


def import_legacy_module(script_path: str):
    script_path = str(Path(script_path).resolve())
    spec = importlib.util.spec_from_file_location("legacy_v18_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_legacy_model(script_path: str, checkpoint_path: str, device, use_ema=True):
    mod = import_legacy_module(script_path)
    model = mod.DualBranchTransformerUNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "G" not in ckpt:
        raise KeyError("Legacy checkpoint missing G")
    model.load_state_dict(ckpt["G"])
    if use_ema:
        shadow = ckpt.get("ema_G")
        if shadow is None:
            raise KeyError("Legacy checkpoint missing ema_G")
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in shadow:
                    p.copy_(shadow[name].to(device=p.device, dtype=p.dtype))
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_legacy_primitives(model, loader, cfg, device, with_lpips=False, support_lock=None):
    metric = MetricComputer(cfg, with_lpips=with_lpips, device=device)
    rows = []
    for batch in loader:
        images = batch["images"].to(device).float()
        cohort_ids = batch["cohort_id"].to(device)
        for cohort in sorted(set(cohort_ids.detach().cpu().tolist())):
            idx = torch.where(cohort_ids == cohort)[0]
            x = images[idx]
            avail = AVAIL_BY_COHORT[int(cohort)]
            for src_tuple, target_idx in primitive_mappings_for_avail(avail):
                task_id = LEGACY_TASK_BY_COHORT_AND_SRC[int(cohort)].get(tuple(src_tuple))
                if task_id is None:
                    continue
                src = torch.tensor(src_tuple, device=device, dtype=torch.float32).view(1,3,1,1).expand(idx.numel(),3,x.shape[-2],x.shape[-1])
                cond = x * src
                tid = torch.full((idx.numel(),), task_id, device=device, dtype=torch.long)
                fake, _, _ = model(cond, src, tid)
                final = cond * src + fake * (1-src)
                pred = final[:,target_idx:target_idx+1]
                gt = x[:,target_idx:target_idx+1]
                vals = metric.compute(pred,gt)
                for local_j, original_j in enumerate(idx.detach().cpu().tolist()):
                    row = {
                        "case_id": batch["case_id"][original_j],
                        "cohort": COHORT_NAMES[int(cohort)],
                        "mapping": mapping_name(src_tuple,target_idx),
                        "psnr": float(vals["psnr"][local_j].cpu()),
                        "ssim": float(vals["ssim"][local_j].cpu()),
                        "mae": float(vals["mae"][local_j].cpu()),
                    }
                    if "lpips" in vals:
                        row["lpips"] = float(vals["lpips"][local_j].cpu())
                    rows.append(row)
    cases = case_aggregate(rows)
    summary = mapping_cohort_balanced_summary(cases, support_lock=support_lock, strict=True)
    return rows, cases, summary
