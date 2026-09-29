#!/usr/bin/env python3
from pathlib import Path
import argparse, json, shutil, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from subsetsynth.config import load_config
from subsetsynth.model import SubsetSynthV2
from subsetsynth.engine import train
from subsetsynth.utils import seed_everything, dump_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/profile.yaml"); args=ap.parse_args(); cfg=load_config(args.config)
    seed_everything(int(cfg["reproducibility"]["seed"]),bool(cfg["reproducibility"].get("deterministic",False)))
    model=SubsetSynthV2(img_size=cfg["model"]["img_size"],token_dim=cfg["model"]["token_dim"],transformer_layers=cfg["model"]["transformer_layers"],transformer_heads=cfg["model"]["transformer_heads"],ff_dim=cfg["model"]["transformer_ff_dim"],physics_enabled=bool(cfg.get("physics",{}).get("enabled",False)))
    run_dir=Path(train(cfg,model,Path(__file__).resolve().parents[1],mode="profile"))
    tr=pd.read_csv(run_dir/"records"/"train_epoch.csv").iloc[-1]; exp=pd.read_csv(run_dir/"records"/"primitive_exposure_detail.csv").fillna(""); weighted=pd.read_csv(run_dir/"records"/"weighted_objective_exposure_detail.csv"); repeat=pd.read_csv(run_dir/"records"/"cohort_repeat_factor_by_epoch.csv").iloc[-1]
    du=shutil.disk_usage(run_dir); checkpoints={p.name:p.stat().st_size for p in (run_dir/"checkpoints").glob("*.pth")}
    report={
        "run_dir":str(run_dir),"epoch_seconds":float(tr["seconds"]),"peak_allocated_gb":float(tr["peak_allocated_bytes"])/2**30,"peak_reserved_gb":float(tr["peak_reserved_bytes"])/2**30,
        "checkpoint_bytes":checkpoints,"filesystem_free_gb_at_report":du.free/2**30,"physics_enabled":bool(tr["physics_enabled"]),
        "sampler_num_samples":int(tr["sampler_num_samples"]),"active_targets_per_sample":float(tr["active_targets_per_sample"]),"decoded_items_per_sample":float(tr["decoded_items_per_sample"]),
        "weighted_mass_per_sample_observed":float(tr["weighted_mass_per_sample_observed"]),"missing_primitives":exp.loc[exp.active_target_count==0,"mapping"].tolist(),
        "primitive_exposure":exp.to_dict("records"),"weighted_objective_exposure":weighted.to_dict("records"),"repeat_factor":repeat.to_dict(),
    }
    dump_json(report,run_dir/"PROFILE_REPORT.json"); print(json.dumps(report,indent=2,ensure_ascii=False)); print(f"PROFILE_RUN={run_dir}")
if __name__=="__main__":main()
