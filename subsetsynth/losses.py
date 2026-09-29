from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Sobel1ch
from .objective import HierarchicalImportanceObjective


class DifferentiableSSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords**2)/(2*sigma*sigma))
        g = g / g.sum()
        kernel = torch.outer(g,g).view(1,1,window_size,window_size)
        self.register_buffer("kernel", kernel)
        self.pad = window_size // 2
        self.data_range = float(data_range)

    def per_image(self, x, y):
        # AMP stability contract: compute SSIM statistics in FP32.
        # Same formula/objective, only safer arithmetic precision.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            y32 = y.float()
            k = self.kernel.to(device=x.device, dtype=torch.float32)
            mu_x = F.conv2d(x32, k, padding=self.pad)
            mu_y = F.conv2d(y32, k, padding=self.pad)
            sx = F.conv2d(x32*x32, k, padding=self.pad) - mu_x*mu_x
            sy = F.conv2d(y32*y32, k, padding=self.pad) - mu_y*mu_y
            sxy = F.conv2d(x32*y32, k, padding=self.pad) - mu_x*mu_y
            c1 = (0.01*self.data_range)**2
            c2 = (0.03*self.data_range)**2
            m = ((2*mu_x*mu_y+c1)*(2*sxy+c2))/((mu_x.square()+mu_y.square()+c1)*(sx+sy+c2)+1e-12)
            return m.flatten(1).mean(1)

    def forward(self, x,y):
        return 1.0 - self.per_image(x,y).mean()


def frequency_l1_per_image(pred, target):
    # AMP stability contract: FFT is evaluated in FP32.
    # Gradients propagate through the float() cast to model outputs.
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        pred32 = pred.float()
        target32 = target.float()
        pf = torch.abs(torch.fft.fft2(pred32, norm="ortho"))
        tf = torch.abs(torch.fft.fft2(target32, norm="ortho"))
        return (pf-tf).abs().flatten(1).mean(1)


class HierarchicalActiveTargetLoss(nn.Module):
    """Returns a *weighted numerator*, not a per-micro-batch mean.

    Engine contract:
      loss_micro = weighted_numerator / EFFECTIVE_BATCH_SIZE
      loss_micro.backward()
      # NO additional division by accumulation steps.

    This makes micro-batches exactly additive across a complete accumulation window.
    """

    def __init__(self, cfg: dict, objective: HierarchicalImportanceObjective):
        super().__init__()
        self.w = cfg["loss"]
        self.objective = objective
        self.ssim = DifferentiableSSIM(
            window_size=int(cfg["evaluation"]["ssim_window"]),
            sigma=float(cfg["evaluation"]["ssim_sigma"]),
        )
        self.sobel = Sobel1ch()

    def _weights(self, src_mask, cohort_id, indices, target_idx, device, dtype):
        vals = []
        for i in indices.detach().cpu().tolist():
            vals.append(self.objective.weight_for(
                src_mask[i].detach().cpu().int().tolist(),
                int(target_idx),
                int(cohort_id[i].detach().cpu().item()),
            ))
        return torch.tensor(vals, device=device, dtype=dtype)

    def forward(
        self,
        preds: Dict[int, dict],
        gt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        cohort_id: torch.Tensor,
    ):
        sums = {
            "l1": gt.new_tensor(0.),
            "ssim": gt.new_tensor(0.),
            "freq": gt.new_tensor(0.),
            "edge": gt.new_tensor(0.),
        }
        n_active = 0
        observed_weight_mass = gt.new_tensor(0.)
        cell_weight_mass: Dict[str, float] = {}

        for t, item in preds.items():
            idx = item["idx"]
            p = item["pred"]
            if idx.numel() == 0:
                continue
            if p.shape[0] != idx.numel():
                raise AssertionError(f"active-only prediction size mismatch for target={t}")
            if not bool(tgt_mask[idx, t].all()):
                raise AssertionError(f"prediction routed to inactive target={t}")
            y = gt.index_select(0, idx)[:, t:t+1]
            weights = self._weights(src_mask, cohort_id, idx, t, p.device, p.dtype)

            per = {
                "l1": (p-y).abs().flatten(1).mean(1),
                "ssim": 1.0 - self.ssim.per_image(p,y),
                "freq": frequency_l1_per_image(p,y),
                "edge": (self.sobel(p)-self.sobel(y)).abs().flatten(1).mean(1),
            }
            for k, values in per.items():
                sums[k] = sums[k] + (weights * values).sum()
            observed_weight_mass = observed_weight_mass + weights.sum()
            n_active += int(idx.numel())

            # Audit only. Mapping identity is never passed to the model.
            for local, original in enumerate(idx.detach().cpu().tolist()):
                from .data import COHORT_NAMES
                from .masks import mapping_name
                m = mapping_name(src_mask[original].detach().cpu().int().tolist(), int(t))
                c = COHORT_NAMES[int(cohort_id[original].detach().cpu().item())]
                key = f"{m}|{c}"
                cell_weight_mass[key] = cell_weight_mass.get(key, 0.0) + float(weights[local].detach().cpu())

        if n_active <= 0:
            raise RuntimeError("No active target items in training micro-batch")

        total_numerator = (
            float(self.w["l1"]) * sums["l1"] +
            float(self.w["ssim"]) * sums["ssim"] +
            float(self.w["frequency"]) * sums["freq"] +
            float(self.w["edge"]) * sums["edge"]
        )
        return total_numerator, sums, n_active, observed_weight_mass, cell_weight_mass


def physics_l1_numerator(recon: torch.Tensor, gt: torch.Tensor, cohort_id: torch.Tensor):
    """Sum of per-sample physics L1 over A/B only; no random micro-batch denominator."""
    active = (cohort_id < 2)
    if not active.any():
        return gt.new_tensor(0.), 0
    per = (recon[active] - gt[active]).abs().flatten(1).mean(1)
    return per.sum(), int(active.sum().item())
