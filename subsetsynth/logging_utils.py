from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

from .config import config_fingerprint, save_resolved_config
from .utils import dump_json, environment_info, package_fingerprint, sha256_file


class RunLogger:
    def __init__(self, cfg: dict, package_root: str | Path, mode: str, resume_run_dir: str | Path | None = None):
        self.cfg = cfg
        self.package_root = Path(package_root).resolve()
        out_root = Path(cfg["paths"]["output_root"])
        run_name = cfg["run"]["name"]

        if resume_run_dir is not None:
            self.run_dir = Path(resume_run_dir).resolve()
            if not self.run_dir.exists():
                raise FileNotFoundError(f"resume run directory not found: {self.run_dir}")
            if self.run_dir.parent.resolve() != out_root.resolve():
                raise RuntimeError(
                    f"resume run directory is outside configured output_root: "
                    f"{self.run_dir} vs {out_root.resolve()}"
                )
            (self.run_dir / "checkpoints").mkdir(exist_ok=True)
            (self.run_dir / "records").mkdir(exist_ok=True)
            (self.run_dir / "evaluation").mkdir(exist_ok=True)
            self.sampler_log_path = self.run_dir / "records" / "sampler_events.jsonl.gz"
            self.events_path = self.run_dir / "records" / "events.jsonl"

            save_resolved_config(cfg, self.run_dir / "config_resolved_resume_recovery.yaml")
            env = environment_info()
            env["package_fingerprint"] = package_fingerprint(self.package_root)
            env["config_fingerprint"] = config_fingerprint(cfg)
            dump_json(env, self.run_dir / "environment_resume_recovery.json")
            self.log_event("run_reopened_for_resume", {
                "mode": mode,
                "run_dir": str(self.run_dir),
            })
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = out_root / f"{run_name}_{mode}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "checkpoints").mkdir()
        (self.run_dir / "records").mkdir()
        (self.run_dir / "evaluation").mkdir()
        self.sampler_log_path = self.run_dir / "records" / "sampler_events.jsonl.gz"
        self.events_path = self.run_dir / "records" / "events.jsonl"
        save_resolved_config(cfg, self.run_dir / "config_resolved.yaml")

        env = environment_info()
        env["package_fingerprint"] = package_fingerprint(self.package_root)
        env["config_fingerprint"] = config_fingerprint(cfg)
        dump_json(env, self.run_dir / "environment.json")
        split_hashes = {}
        for key in ("A","B","C2","C3"):
            p = Path(cfg["data"][key]["split_csv"])
            if p.exists():
                split_hashes[key] = {"path": str(p), "sha256": sha256_file(p)}
        dump_json(split_hashes, self.run_dir / "split_fingerprints.json")

        template = self.package_root / "templates" / "EXPERIMENT_RECORD_TEMPLATE.md"
        if template.exists():
            shutil.copy2(template, self.run_dir / "EXPERIMENT_RECORD.md")
        self.log_event("run_created", {"mode": mode, "run_dir": str(self.run_dir)})

    def log_event(self, kind: str, payload: dict):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def log_sampler_batch(self, epoch: int, global_step: int, batch: dict, src_mask, tgt_mask):
        if not self.cfg["logging"].get("log_sampler_events", True):
            return
        rows = []
        cohorts = batch["cohort_id"].detach().cpu().tolist()
        avails = batch["avail_mask"].detach().cpu().int().tolist()
        srcs = src_mask.detach().cpu().int().tolist()
        tgts = tgt_mask.detach().cpu().int().tolist()
        for case_id, cohort, avail, src, tgt in zip(batch["case_id"], cohorts, avails, srcs, tgts):
            rows.append({
                "seed": self.cfg["reproducibility"]["seed"],
                "epoch": epoch,
                "global_step": global_step,
                "case_id": case_id,
                "cohort_id": cohort,
                "avail_mask": avail,
                "src_mask": src,
                "tgt_mask": tgt,
            })
        with gzip.open(self.sampler_log_path, "at", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
