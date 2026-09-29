#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from subsetsynth.losses import DifferentiableSSIM, frequency_l1_per_image

def main():
    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] CUDA not available")
    device = torch.device("cuda")
    torch.manual_seed(20260822)

    p = torch.rand(2, 1, 512, 512, device=device, requires_grad=True)
    y = torch.rand(2, 1, 512, 512, device=device)
    ssim = DifferentiableSSIM().to(device)

    with torch.amp.autocast("cuda", enabled=True):
        p_amp = p * 1.0
        f = frequency_l1_per_image(p_amp, y)
        s = 1.0 - ssim.per_image(p_amp, y)
        loss = f.mean() + s.mean()

    print("frequency dtype:", f.dtype, "finite:", bool(torch.isfinite(f).all()))
    print("ssim_loss dtype:", s.dtype, "finite:", bool(torch.isfinite(s).all()))
    print("combined finite:", bool(torch.isfinite(loss)))
    loss.backward()
    print("grad finite:", bool(torch.isfinite(p.grad).all()))
    print("grad max abs:", float(p.grad.abs().max()))

    ok = bool(torch.isfinite(f).all() and torch.isfinite(s).all()
              and torch.isfinite(loss) and torch.isfinite(p.grad).all())
    if not ok:
        raise SystemExit("[FAIL] AMP loss stability smoke test")
    print("[PASS] AMP loss stability smoke test")

if __name__ == "__main__":
    main()
