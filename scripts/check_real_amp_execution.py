#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from subsetsynth.config import load_config
from subsetsynth.data import MultiContrastNPYDataset, make_train_loader, paired_training_augment
from subsetsynth.ema import EMA
from subsetsynth.engine import build_optimizer
from subsetsynth.locks import load_support_lock, load_semantics_lock
from subsetsynth.losses import HierarchicalActiveTargetLoss
from subsetsynth.masks import SubsetBagManager
from subsetsynth.model import SubsetSynthV2
from subsetsynth.objective import HierarchicalImportanceObjective
from subsetsynth.routing import decode_active_targets
from subsetsynth.utils import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/profile.yaml")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--init-scale", type=float, default=None,
                    help="Diagnostic override only; does not edit YAML.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.init_scale is not None:
        cfg.setdefault("amp", {})["init_scale"] = float(args.init_scale)

    load_semantics_lock(cfg)
    support_lock = load_support_lock(cfg)
    objective = HierarchicalImportanceObjective.from_config(cfg, support_lock)

    if not torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA unavailable")
    device = torch.device("cuda")

    seed_everything(
        int(cfg["reproducibility"]["seed"]),
        bool(cfg["reproducibility"].get("deterministic", False)),
    )

    micro = int(cfg["training"]["micro_batch_size"])
    effective = int(cfg["training"]["effective_batch_size"])
    if effective % micro != 0:
        raise SystemExit("[FAIL] invalid accumulation geometry")
    accum = effective // micro

    amp_cfg = cfg.get("amp", {})
    amp_enabled = bool(amp_cfg.get("enabled", True))
    init_scale = float(amp_cfg.get("init_scale", 8192.0))

    model = SubsetSynthV2(
        img_size=cfg["model"]["img_size"],
        token_dim=cfg["model"]["token_dim"],
        transformer_layers=cfg["model"]["transformer_layers"],
        transformer_heads=cfg["model"]["transformer_heads"],
        ff_dim=cfg["model"]["transformer_ff_dim"],
        physics_enabled=bool(cfg.get("physics", {}).get("enabled", False)),
    ).to(device)

    loss_fn = HierarchicalActiveTargetLoss(cfg, objective).to(device)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=init_scale,
        growth_factor=float(amp_cfg.get("growth_factor", 2.0)),
        backoff_factor=float(amp_cfg.get("backoff_factor", 0.5)),
        growth_interval=int(amp_cfg.get("growth_interval", 2000)),
    )
    ema = EMA(model, decay=float(cfg["training"]["ema_decay"]))
    bags = SubsetBagManager(seed=int(cfg["reproducibility"]["seed"]) + 4242)

    ds = MultiContrastNPYDataset(cfg, "train")
    loader = make_train_loader(ds, cfg, epoch=1)

    print(
        f"dataset_slices={len(ds)} loader_microbatches={len(loader)} "
        f"micro={micro} accum={accum} effective={effective} init_scale={init_scale:g}"
    )

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    completed = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].to(device, non_blocking=True).float()
        avail = batch["avail_mask"].to(device, non_blocking=True).bool()
        cohort = batch["cohort_id"].to(device, non_blocking=True)

        if not torch.isfinite(images).all():
            raise FloatingPointError(f"[FAIL] nonfinite raw input at microbatch {batch_idx}")

        src_mask, tgt_mask = bags.sample_batch(cohort, avail)
        x_aug = paired_training_augment(images, avail, src_mask, cfg)
        if not torch.isfinite(x_aug).all():
            raise FloatingPointError(f"[FAIL] nonfinite augmented input at microbatch {batch_idx}")

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            z, skips = model.encode(x_aug, src_mask)
            preds = decode_active_targets(model, z, skips, tgt_mask)
            recon_num, comp_num, n_active, _, _ = loss_fn(
                preds, x_aug, src_mask, tgt_mask, cohort
            )
            total_scaled = recon_num / effective

        if not bool(torch.isfinite(total_scaled)):
            raise FloatingPointError(f"[FAIL] nonfinite loss at microbatch {batch_idx}")
        if not all(bool(torch.isfinite(v)) for v in comp_num.values()):
            raise FloatingPointError(f"[FAIL] nonfinite loss component at microbatch {batch_idx}")
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError(f"[FAIL] nonfinite latent at microbatch {batch_idx}")
        if not all(bool(torch.isfinite(v["pred"]).all()) for v in preds.values()):
            raise FloatingPointError(f"[FAIL] nonfinite prediction at microbatch {batch_idx}")

        scaler.scale(total_scaled).backward()
        print(f"micro={batch_idx:02d} active={n_active} loss_scaled={float(total_scaled.detach()):.6f}")

        if batch_idx % accum == 0:
            scale_before = float(scaler.get_scale())
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["optimizer"]["grad_clip_norm"]),
                error_if_nonfinite=False,
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"[FAIL] nonfinite gradient norm at optimizer window "
                    f"{completed + 1}; init_scale={init_scale:g}"
                )

            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if scale_after < scale_before:
                raise FloatingPointError(
                    f"[FAIL] GradScaler overflow/step skip: {scale_before:g}->{scale_after:g}"
                )

            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError("[FAIL] nonfinite parameter after optimizer step")

            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            completed += 1
            print(
                f"  [STEP {completed}] grad_norm={float(grad_norm):.6f} "
                f"scaler={scale_before:g}->{scale_after:g}"
            )

            if completed >= args.steps:
                break

    print(f"peak_allocated_gb={torch.cuda.max_memory_allocated()/2**30:.3f}")
    print(f"peak_reserved_gb={torch.cuda.max_memory_reserved()/2**30:.3f}")
    print(f"optimizer_steps={completed}")
    if completed != args.steps:
        raise RuntimeError("[FAIL] smoke test ended before requested optimizer steps")
    print("[PASS] REAL training-path AMP execution smoke test")


if __name__ == "__main__":
    main()
