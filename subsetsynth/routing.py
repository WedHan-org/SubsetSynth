from __future__ import annotations

from typing import Dict
import torch


def decode_active_targets(model, z: torch.Tensor, skips, tgt_mask: torch.Tensor) -> Dict[int, dict]:
    """Decode only samples that actively request each target.

    Returns {target_idx: {"idx": LongTensor, "pred": Tensor[N_active,1,H,W]}}.
    This avoids constructing decoder/RRDB graphs for inactive samples.
    """
    preds: Dict[int, dict] = {}
    for t in range(tgt_mask.shape[1]):
        idx = torch.where(tgt_mask[:, t])[0]
        if idx.numel() == 0:
            continue
        z_t = z.index_select(0, idx)
        skips_t = [s.index_select(0, idx) for s in skips]
        tid = torch.full((idx.numel(),), int(t), device=z.device, dtype=torch.long)
        pred = model.decode(z_t, skips_t, tid)
        preds[int(t)] = {"idx": idx, "pred": pred}
    return preds
