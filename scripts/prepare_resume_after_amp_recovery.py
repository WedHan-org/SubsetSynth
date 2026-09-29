#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
import torch

def backup_once(path: Path) -> Path:
    backup = path.with_name(path.name + ".pre_amp_recovery_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup

def clean_csv_after_epoch(path: Path, checkpoint_epoch: int) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            return 0
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    kept, removed = [], 0
    for row in rows:
        try:
            e = int(float(row["epoch"]))
        except Exception:
            kept.append(row); continue
        if e <= checkpoint_epoch: kept.append(row)
        else: removed += 1
    if removed:
        backup_once(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(kept)
        os.replace(tmp, path)
    return removed

def clean_sampler_log(path: Path, checkpoint_epoch: int):
    if not path.exists():
        return 0, 0
    backup_once(path)
    kept = removed = 0
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(path, "rt", encoding="utf-8") as src, gzip.open(tmp, "wt", encoding="utf-8") as dst:
        for line in src:
            try:
                e = int(json.loads(line).get("epoch", -1))
            except Exception:
                dst.write(line); kept += 1; continue
            if e <= checkpoint_epoch:
                dst.write(line); kept += 1
            else:
                removed += 1
    os.replace(tmp, path)
    return kept, removed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True)
    args = ap.parse_args()
    ckpt_path = Path(args.resume).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"[ERROR] checkpoint not found: {ckpt_path}")
    if ckpt_path.parent.name != "checkpoints":
        raise SystemExit("[ERROR] checkpoint must be under <run_dir>/checkpoints/")
    run_dir = ckpt_path.parent.parent
    records = run_dir / "records"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "SubsetSynth_v2_0b_resume":
        raise SystemExit(f"[ERROR] unexpected checkpoint format: {ckpt.get('format')}")
    epoch = int(ckpt["epoch"])
    global_step = int(ckpt["global_step"])
    scaler_state = ckpt.get("scaler", {})
    scale = scaler_state.get("scale", scaler_state.get("_scale", None))
    if isinstance(scale, torch.Tensor): scale = float(scale.item())
    elif scale is not None: scale = float(scale)

    print(f"[CHECK] run_dir={run_dir}")
    print(f"[CHECK] checkpoint_epoch={epoch}")
    print(f"[CHECK] resume_start_epoch={epoch + 1}")
    print(f"[CHECK] global_step={global_step}")
    print(f"[CHECK] restored_amp_scale={scale}")
    if scale is not None and not (1024.0 <= scale <= 8192.0):
        raise SystemExit(f"[ERROR] restored scaler scale {scale} outside [1024,8192]")

    kept, removed = clean_sampler_log(records / "sampler_events.jsonl.gz", epoch)
    print(f"[CLEAN] sampler rows kept={kept} removed_post_checkpoint={removed}")
    csv_removed = {}
    for path in sorted(records.glob("*.csv")):
        n = clean_csv_after_epoch(path, epoch)
        if n: csv_removed[path.name] = n
    print(f"[CLEAN] csv_rows_removed={csv_removed}")

    note = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "action": "prepare_same_run_resume_after_recoverable_amp_overflow",
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": epoch,
        "resume_start_epoch": epoch + 1,
        "global_step": global_step,
        "restored_amp_scale": scale,
        "sampler_rows_removed_after_checkpoint": removed,
        "csv_rows_removed_after_checkpoint": csv_removed,
        "events_jsonl_preserved": True,
    }
    p = run_dir / "AMP_RECOVERY_RESUME_NOTE.json"
    p.write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"[OK] wrote {p}")
    print("[PASS] resume directory cleaned to checkpoint boundary")

if __name__ == "__main__":
    main()
