from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .data import (
    make_train_loader,
    make_eval_loader,
    paired_training_augment,
    MultiContrastNPYDataset,
    COHORT_NAMES,
    training_sampler_plan,
)
from .ema import EMA
from .evaluator import evaluate_primitives
from .logging_utils import RunLogger, append_csv
from .losses import HierarchicalActiveTargetLoss, physics_l1_numerator
from .locks import load_support_lock, load_semantics_lock
from .masks import SubsetBagManager, mapping_name
from .objective import HierarchicalImportanceObjective
from .physics import DynamicBlochLayer
from .routing import decode_active_targets
from .utils import atomic_torch_save, dump_json, sha256_file


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "emb" in name or "role_" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": float(cfg["optimizer"]["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=float(cfg["optimizer"]["lr"]), betas=tuple(cfg["optimizer"]["betas"]))


def _selection_better(psnr, ssim, best_psnr, best_ssim, tie_db=0.05):
    if best_psnr is None:
        return True
    if psnr > best_psnr + tie_db:
        return True
    if abs(psnr - best_psnr) < tie_db and ssim > best_ssim:
        return True
    return False


def save_best_checkpoint(path, model, ema, epoch, global_step, summary, cfg, objective_dict):
    payload = {
        "format": "SubsetSynth_v2_0b_best",
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "selection_summary": summary,
        "objective": objective_dict,
        "physics_enabled": bool(cfg.get("physics", {}).get("enabled", False)),
        "config_path": cfg.get("_config_path"),
    }
    atomic_torch_save(payload, path)


def save_resume_checkpoint(path, model, ema, optimizer, scheduler, scaler, subset_bags, epoch, global_step, best_state, cfg, objective_dict):
    payload = {
        "format": "SubsetSynth_v2_0b_resume",
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "subset_bags": subset_bags.state_dict(),
        "best_state": best_state,
        "objective": objective_dict,
        "physics_enabled": bool(cfg.get("physics", {}).get("enabled", False)),
        "config_path": cfg.get("_config_path"),
    }
    atomic_torch_save(payload, path)


def load_resume(path, model, ema, optimizer, scheduler, scaler, subset_bags, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    subset_bags.load_state_dict(ckpt["subset_bags"])
    return int(ckpt["epoch"]) + 1, int(ckpt["global_step"]), ckpt.get("best_state", {})


def train(cfg: dict, model, package_root: str | Path, mode: str = "train", resume: str | None = None):
    # Hard provenance gates before any clinical NPY is opened.
    semantic_lock = load_semantics_lock(cfg)
    support_lock = load_support_lock(cfg)
    objective = HierarchicalImportanceObjective.from_config(cfg, support_lock)

    physics_enabled = bool(cfg.get("physics", {}).get("enabled", False))
    if bool(getattr(model, "physics_enabled", False)) != physics_enabled:
        raise RuntimeError("Model physics_enabled does not match config physics.enabled")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    amp_cfg = cfg.get("amp", {})
    amp_enabled = bool(amp_cfg.get("enabled", True)) and device.type == "cuda"
    amp_init_scale = float(amp_cfg.get("init_scale", 8192.0))
    amp_growth_factor = float(amp_cfg.get("growth_factor", 2.0))
    amp_backoff_factor = float(amp_cfg.get("backoff_factor", 0.5))
    amp_growth_interval = int(amp_cfg.get("growth_interval", 2000))
    amp_fail_on_overflow = bool(amp_cfg.get("fail_on_overflow", False))
    amp_recover_overflow = bool(amp_cfg.get("recover_overflow", True))
    amp_max_consecutive_overflows = int(amp_cfg.get("max_consecutive_overflows", 3))
    amp_min_scale = float(amp_cfg.get("min_scale", 1024.0))

    if amp_init_scale <= 0:
        raise ValueError("amp.init_scale must be >0")
    if amp_growth_factor <= 1.0:
        raise ValueError("amp.growth_factor must be >1")
    if not (0.0 < amp_backoff_factor < 1.0):
        raise ValueError("amp.backoff_factor must be in (0,1)")
    if amp_growth_interval < 1:
        raise ValueError("amp.growth_interval must be >=1")
    if amp_max_consecutive_overflows < 1:
        raise ValueError("amp.max_consecutive_overflows must be >=1")
    if amp_min_scale <= 0:
        raise ValueError("amp.min_scale must be >0")

    resume_run_dir = None
    if resume:
        resume_path = Path(resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        if resume_path.parent.name != "checkpoints":
            raise RuntimeError("resume checkpoint must live under <run_dir>/checkpoints/")
        resume_run_dir = resume_path.parent.parent

    logger = RunLogger(cfg, package_root, mode, resume_run_dir=resume_run_dir)
    logger.log_event("amp_contract", {
        "enabled": amp_enabled,
        "init_scale": amp_init_scale,
        "growth_factor": amp_growth_factor,
        "backoff_factor": amp_backoff_factor,
        "growth_interval": amp_growth_interval,
        "fail_on_overflow": amp_fail_on_overflow,
        "recover_overflow": amp_recover_overflow,
        "max_consecutive_overflows": amp_max_consecutive_overflows,
        "min_scale": amp_min_scale,
    })
    dump_json(objective.to_dict(), logger.run_dir / "objective_frozen.json")
    dump_json(support_lock, logger.run_dir / "mapping_cohort_support_lock_used.json")
    dump_json(semantic_lock, logger.run_dir / "npy_semantics_lock_used.json")

    train_ds = MultiContrastNPYDataset(cfg, "train")
    val_ds = MultiContrastNPYDataset(cfg, "val")
    val_loader = make_eval_loader(val_ds, cfg)
    sampler_plan = training_sampler_plan(train_ds, cfg)
    dump_json({k:v for k,v in sampler_plan.items() if k != "weights"}, logger.run_dir / "training_sampler_plan.json")

    micro_batch = int(cfg["training"]["micro_batch_size"])
    effective_batch = int(cfg["training"]["effective_batch_size"])
    accum_steps = effective_batch // micro_batch
    if effective_batch != micro_batch * accum_steps:
        raise AssertionError("effective_batch_size != micro_batch_size * accum_steps")
    if sampler_plan["num_samples"] % effective_batch != 0:
        raise AssertionError("sampler_num_samples must be a complete effective-batch multiple")

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["training"]["max_epochs"]), eta_min=float(cfg["optimizer"]["lr"]) * 0.1
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=amp_init_scale,
        growth_factor=amp_growth_factor,
        backoff_factor=amp_backoff_factor,
        growth_interval=amp_growth_interval,
    )
    ema = EMA(model, decay=float(cfg["training"]["ema_decay"]))
    bloch = DynamicBlochLayer().to(device) if physics_enabled else None
    recon_loss = HierarchicalActiveTargetLoss(cfg, objective).to(device)
    subset_bags = SubsetBagManager(seed=int(cfg["reproducibility"]["seed"]) + 4242)

    physics_expected_active = float(cfg["sampling"]["cohort_proposal"]["A"]) + float(cfg["sampling"]["cohort_proposal"]["B"])
    if physics_enabled and physics_expected_active <= 0:
        raise RuntimeError("physics enabled but proposal has zero A/B probability")

    start_epoch, global_step = 1, 0
    best_state = {"psnr": None, "ssim": -float("inf"), "epoch": None, "bad_events": 0, "progress_psnr": -float("inf")}
    if resume:
        start_epoch, global_step, restored = load_resume(
            resume, model, ema, optimizer, scheduler, scaler, subset_bags, device
        )
        best_state.update(restored)
        logger.log_event("resumed", {
            "checkpoint": resume,
            "start_epoch": start_epoch,
            "global_step": global_step,
            "restored_amp_scale": float(scaler.get_scale()),
            "same_run_directory": True,
        })

    max_epochs = int(cfg["training"]["max_epochs"])
    val_interval = int(cfg["training"]["val_interval"])
    min_epochs = int(cfg["training"]["min_epochs"])
    patience = int(cfg["training"]["patience_validation_events"])
    min_delta = float(cfg["training"]["min_delta_psnr_db"])
    tie_db = float(cfg["checkpoint"]["ssim_tie_psnr_db"])
    use_early = bool(cfg["training"].get("early_stopping", True))

    latest_path = logger.run_dir / "checkpoints" / "latest_resume.pth"
    best_path = logger.run_dir / "checkpoints" / "best_mapping_balanced_psnr.pth"
    objective_dict = objective.to_dict()

    consecutive_amp_overflows = 0
    total_amp_overflows = 0

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        if bloch is not None:
            bloch.train()
        loader = make_train_loader(train_ds, cfg, epoch)
        if len(loader) % accum_steps != 0:
            raise AssertionError("partial accumulation window detected")
        optimizer.zero_grad(set_to_none=True)

        epoch_exposure = Counter()
        exposure_by_cohort = Counter()
        weighted_mass_by_cell = Counter()
        cohort_sample_counts = Counter()
        unique_paths = {c:set() for c in ("A","B","C2","C3")}
        total_recon_numerator = 0.0
        component_numerators = Counter()
        total_observed_weight_mass = 0.0
        total_active_targets = 0
        total_physics_numerator = 0.0
        total_physics_active = 0
        n_micro_batches = 0
        optimizer_steps_epoch = 0
        amp_overflow_skips_epoch = 0
        decoded_items = 0
        max_grad_norm_preclip = 0.0
        amp_scale_min = float("inf")
        amp_scale_max = 0.0
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        loop = tqdm(loader, desc=f"train e{epoch}")
        for batch_idx, batch in enumerate(loop, start=1):
            images = batch["images"].to(device, non_blocking=True).float()
            avail = batch["avail_mask"].to(device, non_blocking=True).bool()
            cohort = batch["cohort_id"].to(device, non_blocking=True)
            if images.shape[0] != micro_batch:
                raise AssertionError(f"partial micro-batch: got {images.shape[0]}, expected {micro_batch}")

            src_mask, tgt_mask = subset_bags.sample_batch(cohort, avail)
            logger.log_sampler_batch(epoch, global_step, batch, src_mask, tgt_mask)
            for b in range(images.shape[0]):
                c_name = COHORT_NAMES[int(cohort[b].item())]
                cohort_sample_counts[c_name] += 1
                unique_paths[c_name].add(str(batch["path"][b]))
                for t in range(3):
                    if bool(tgt_mask[b,t]):
                        mname = mapping_name(src_mask[b].int().cpu().tolist(), t)
                        epoch_exposure[mname] += 1
                        exposure_by_cohort[(mname, c_name)] += 1

            x_aug = paired_training_augment(images, avail, src_mask, cfg)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                z, skips = model.encode(x_aug, src_mask)
                preds = decode_active_targets(model, z, skips, tgt_mask)
                decoded_items += sum(int(item["idx"].numel()) for item in preds.values())
                recon_num, comp_num, n_active, observed_weight_mass, cell_mass = recon_loss(
                    preds, x_aug, src_mask, tgt_mask, cohort
                )
                # Fixed denominator over a *complete effective batch*.
                # IMPORTANT: there is deliberately no /accum_steps anywhere below.
                total_scaled = recon_num / effective_batch

                phys_num = x_aug.new_tensor(0.)
                phys_n = 0
                if physics_enabled:
                    phys_maps = model.physics_maps(z)
                    phys_recon = bloch(phys_maps, cohort, jitter=True)
                    phys_num, phys_n = physics_l1_numerator(phys_recon, x_aug, cohort)
                    total_scaled = total_scaled + float(cfg["physics"]["weight"]) * (
                        phys_num / (effective_batch * physics_expected_active)
                    )

            if not torch.isfinite(total_scaled):
                logger.log_event("fatal_nonfinite_loss", {
                    "epoch": epoch, "global_step": global_step, "loss_scaled": float(total_scaled.detach().cpu())
                })
                raise FloatingPointError("NaN/Inf loss detected")

            scaler.scale(total_scaled).backward()
            if batch_idx % accum_steps == 0:
                scale_before = float(scaler.get_scale())
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["optimizer"]["grad_clip_norm"]),
                    error_if_nonfinite=False,
                )

                if not torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    optimizer.zero_grad(set_to_none=True)

                    amp_overflow_skips_epoch += 1
                    total_amp_overflows += 1
                    consecutive_amp_overflows += 1
                    amp_scale_min = min(amp_scale_min, scale_after)
                    amp_scale_max = max(amp_scale_max, scale_before)

                    logger.log_event("amp_overflow_recovered", {
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_idx": batch_idx,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                        "grad_norm": str(float(grad_norm.detach().cpu())),
                        "consecutive_overflows": consecutive_amp_overflows,
                        "total_overflows": total_amp_overflows,
                        "optimizer_step_counted": False,
                        "ema_updated": False,
                    })

                    if (
                        (not amp_recover_overflow)
                        or amp_fail_on_overflow
                        or consecutive_amp_overflows >= amp_max_consecutive_overflows
                        or scale_after < amp_min_scale
                    ):
                        logger.log_event("fatal_amp_overflow_policy", {
                            "epoch": epoch,
                            "global_step": global_step,
                            "batch_idx": batch_idx,
                            "scale_after": scale_after,
                            "consecutive_overflows": consecutive_amp_overflows,
                            "total_overflows": total_amp_overflows,
                        })
                        raise FloatingPointError(
                            f"AMP overflow exceeded recovery policy "
                            f"(epoch={epoch}, batch_idx={batch_idx}, "
                            f"scale={scale_before:g}->{scale_after:g}, "
                            f"consecutive={consecutive_amp_overflows})"
                        )
                else:
                    grad_norm_value = float(grad_norm.detach().cpu())
                    max_grad_norm_preclip = max(max_grad_norm_preclip, grad_norm_value)

                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())

                    if amp_enabled and scale_after < scale_before:
                        optimizer.zero_grad(set_to_none=True)
                        amp_overflow_skips_epoch += 1
                        total_amp_overflows += 1
                        consecutive_amp_overflows += 1
                        amp_scale_min = min(amp_scale_min, scale_after)
                        amp_scale_max = max(amp_scale_max, scale_before)
                        logger.log_event("amp_overflow_recovered", {
                            "epoch": epoch,
                            "global_step": global_step,
                            "batch_idx": batch_idx,
                            "scale_before": scale_before,
                            "scale_after": scale_after,
                            "grad_norm": grad_norm_value,
                            "consecutive_overflows": consecutive_amp_overflows,
                            "total_overflows": total_amp_overflows,
                            "optimizer_step_counted": False,
                            "ema_updated": False,
                        })
                        if (
                            (not amp_recover_overflow)
                            or amp_fail_on_overflow
                            or consecutive_amp_overflows >= amp_max_consecutive_overflows
                            or scale_after < amp_min_scale
                        ):
                            raise FloatingPointError(
                                f"AMP overflow exceeded recovery policy "
                                f"(epoch={epoch}, batch_idx={batch_idx}, "
                                f"scale={scale_before:g}->{scale_after:g}, "
                                f"consecutive={consecutive_amp_overflows})"
                            )
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        consecutive_amp_overflows = 0
                        ema.update(model)
                        global_step += 1
                        optimizer_steps_epoch += 1
                        amp_scale_min = min(amp_scale_min, scale_after)
                        amp_scale_max = max(amp_scale_max, scale_after)

            total_recon_numerator += float(recon_num.detach().cpu())
            for k,v in comp_num.items():
                component_numerators[k] += float(v.detach().cpu())
            total_observed_weight_mass += float(observed_weight_mass.detach().cpu())
            total_active_targets += int(n_active)
            total_physics_numerator += float(phys_num.detach().cpu())
            total_physics_active += int(phys_n)
            for key, mass in cell_mass.items():
                weighted_mass_by_cell[key] += float(mass)
            n_micro_batches += 1
            running = total_recon_numerator / max(batch_idx * micro_batch, 1)
            loop.set_postfix(obj=f"{running:.4f}")

        planned_windows = sampler_plan["num_samples"] // effective_batch
        if optimizer_steps_epoch + amp_overflow_skips_epoch != planned_windows:
            raise AssertionError(
                "successful optimizer steps + AMP-skipped windows "
                "does not equal complete effective-batch plan"
            )

        scheduler.step()
        elapsed = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        n_samples = sampler_plan["num_samples"]
        train_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "n_slice_records_dataset": sampler_plan["n_records"],
            "sampler_num_samples": n_samples,
            "micro_batch_size": micro_batch,
            "accum_steps": accum_steps,
            "effective_batch_size": effective_batch,
            "optimizer_steps": optimizer_steps_epoch,
            "planned_optimizer_windows": planned_windows,
            "amp_overflow_skips": amp_overflow_skips_epoch,
            "amp_overflows_total_so_far": total_amp_overflows,
            "amp_enabled": amp_enabled,
            "amp_init_scale": amp_init_scale,
            "amp_scale_min": amp_scale_min if planned_windows > 0 else float(scaler.get_scale()),
            "amp_scale_max": amp_scale_max if planned_windows > 0 else float(scaler.get_scale()),
            "max_grad_norm_preclip": max_grad_norm_preclip,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
            "objective_recon_estimate": total_recon_numerator / n_samples,
            "weighted_mass_per_sample_observed": total_observed_weight_mass / n_samples,
            "active_targets_per_sample": total_active_targets / n_samples,
            "decoded_items_per_sample": decoded_items / n_samples,
            "loss_l1_weighted_estimate": component_numerators["l1"] / n_samples,
            "loss_ssim_weighted_estimate": component_numerators["ssim"] / n_samples,
            "loss_freq_weighted_estimate": component_numerators["freq"] / n_samples,
            "loss_edge_weighted_estimate": component_numerators["edge"] / n_samples,
            "physics_enabled": physics_enabled,
            "physics_conditional_l1_estimate": (
                total_physics_numerator / (n_samples * physics_expected_active) if physics_enabled else 0.0
            ),
        }
        append_csv(logger.run_dir / "records" / "train_epoch.csv", train_row)

        canonical_mappings = list(objective.mapping_order)
        exp_row = {"epoch": epoch, **{f"exposure_{m}": int(epoch_exposure.get(m,0)) for m in canonical_mappings}}
        append_csv(logger.run_dir / "records" / "primitive_exposure_by_epoch.csv", exp_row)
        total_targets = max(sum(epoch_exposure.values()), 1)
        detail_rows = []
        weighted_rows = []
        for m in canonical_mappings:
            total_m = int(epoch_exposure.get(m,0))
            supporters = support_lock["mapping_support"][m]
            detail_rows.append({
                "epoch": epoch,
                "mapping": m,
                "active_target_count": total_m,
                "active_target_pixels": total_m * int(cfg["data"]["img_size"])**2,
                "fraction_all_active_targets": total_m / total_targets,
                "cohort_contributions": ";".join(f"{c}:{exposure_by_cohort.get((m,c),0)}" for c in supporters),
            })
            for c in supporters:
                cell = objective.cell(m,c)
                observed_mass = float(weighted_mass_by_cell.get(f"{m}|{c}",0.0)) / n_samples
                weighted_rows.append({
                    "epoch": epoch,
                    "mapping": m,
                    "cohort": c,
                    "raw_count": int(exposure_by_cohort.get((m,c),0)),
                    "proposal_q": cell.q,
                    "target_alpha": cell.alpha,
                    "importance_weight": cell.weight,
                    "observed_weighted_mass_per_sample": observed_mass,
                    "observed_minus_target": observed_mass - cell.alpha,
                })
        for fname, frame in (
            ("primitive_exposure_detail.csv", pd.DataFrame(detail_rows)),
            ("weighted_objective_exposure_detail.csv", pd.DataFrame(weighted_rows)),
        ):
            path = logger.run_dir / "records" / fname
            frame.to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)

        repeat_row = {"epoch": epoch}
        cohort_row = {"epoch": epoch}
        for c in ("A","B","C2","C3"):
            draws = int(cohort_sample_counts.get(c,0))
            uniq = len(unique_paths[c])
            cohort_row[f"samples_{c}"] = draws
            repeat_row[f"draws_{c}"] = draws
            repeat_row[f"unique_slice_records_{c}"] = uniq
            repeat_row[f"repeat_factor_{c}"] = draws / max(uniq,1)
        append_csv(logger.run_dir / "records" / "cohort_sample_counts_by_epoch.csv", cohort_row)
        append_csv(logger.run_dir / "records" / "cohort_repeat_factor_by_epoch.csv", repeat_row)

        missing_primitives = [m for m in canonical_mappings if epoch_exposure.get(m,0)==0]
        logger.log_event("objective_exposure_audit", {
            "epoch":epoch,
            "missing_primitives":missing_primitives,
            "total_active_targets":sum(epoch_exposure.values()),
            "weighted_mass_per_sample":total_observed_weight_mass/n_samples,
            "expected_weighted_mass_per_sample":objective.expected_mass_per_sample,
        })

        do_val = (epoch % val_interval == 0) or epoch == max_epochs or mode == "profile"
        summary = None
        if do_val:
            ema.apply_to(model)
            try:
                _, case_rows, summary = evaluate_primitives(
                    model, val_loader, cfg, device, with_lpips=False, desc=f"val e{epoch}", support_lock=support_lock
                )
                overall = summary["overall_mapping_balanced"]
                psnr, ssim = float(overall["psnr"]), float(overall["ssim"])
                append_csv(logger.run_dir / "records" / "validation_events.csv", {
                    "epoch": epoch,
                    "mapping_balanced_psnr": psnr,
                    "mapping_balanced_ssim": ssim,
                    "mapping_balanced_mae": float(overall.get("mae", float("nan"))),
                })
                flat = [{"epoch":epoch,"mapping":m,**vals} for m,vals in summary["mapping"].items()]
                pd.DataFrame(flat).to_csv(logger.run_dir / "evaluation" / f"val_epoch_{epoch:03d}_mapping_summary.csv", index=False)
                for c, vals in summary["cohort_mapping_balanced"].items():
                    append_csv(logger.run_dir / "records" / "validation_by_cohort.csv", {"epoch":epoch,"cohort":c,**vals})

                if _selection_better(psnr, ssim, best_state["psnr"], best_state["ssim"], tie_db=tie_db):
                    best_state.update({"psnr": psnr, "ssim": ssim, "epoch": epoch})
                    save_best_checkpoint(best_path, model, ema, epoch, global_step, summary, cfg, objective_dict)
                    logger.log_event("best_checkpoint_updated", {"epoch": epoch, "psnr": psnr, "ssim": ssim})

                if psnr > best_state["progress_psnr"] + min_delta:
                    best_state["progress_psnr"] = psnr
                    best_state["bad_events"] = 0
                elif epoch >= min_epochs:
                    best_state["bad_events"] += 1
                logger.log_event("validation", {"epoch": epoch, "psnr": psnr, "ssim": ssim, "bad_events": best_state["bad_events"]})
            finally:
                ema.restore(model)

        save_resume_checkpoint(
            latest_path, model, ema, optimizer, scheduler, scaler, subset_bags,
            epoch, global_step, best_state, cfg, objective_dict
        )
        if do_val and summary is not None and epoch in [int(x) for x in cfg.get("checkpoint", {}).get("keep_milestones", [])]:
            ema.apply_to(model)
            try:
                milestone = logger.run_dir / "checkpoints" / f"milestone_epoch_{epoch:03d}.pth"
                save_best_checkpoint(milestone, model, ema, epoch, global_step, summary, cfg, objective_dict)
            finally:
                ema.restore(model)
            logger.log_event("milestone_checkpoint", {"epoch":epoch,"path":str(milestone)})

        if do_val and use_early and epoch >= min_epochs and best_state["bad_events"] >= patience:
            logger.log_event("early_stop", {"epoch": epoch, "bad_events": best_state["bad_events"]})
            break
        if mode == "profile":
            break

    manifest = {
        "run_dir": str(logger.run_dir),
        "scientific_objective": "9 mappings equal; supporting cohorts equal within each mapping",
        "proposal_is_fairness_mechanism": False,
        "objective_expected_mass_per_sample": objective.expected_mass_per_sample,
        "train_slice_records": sampler_plan["n_records"],
        "sampler_num_samples_per_epoch": sampler_plan["num_samples"],
        "micro_batch_size": micro_batch,
        "accum_steps": accum_steps,
        "effective_batch_size": effective_batch,
        "physics_enabled": physics_enabled,
        "best_epoch": best_state.get("epoch"),
        "best_psnr": best_state.get("psnr"),
        "best_ssim": best_state.get("ssim"),
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "best_checkpoint_sha256": sha256_file(best_path) if best_path.exists() else None,
        "latest_checkpoint_sha256": sha256_file(latest_path) if latest_path.exists() else None,
    }
    dump_json(manifest, logger.run_dir / "RUN_MANIFEST.json")
    return logger.run_dir
