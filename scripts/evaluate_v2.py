#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from subsetsynth.config import load_config
from subsetsynth.model import SubsetSynthV2
from subsetsynth.checkpoint import load_v2_for_inference
from subsetsynth.data import MultiContrastNPYDataset, make_eval_loader
from subsetsynth.evaluator import evaluate_primitives, save_evaluation
from subsetsynth.matrix_eval import evaluate_external_matrix
from subsetsynth.utils import dump_json, sha256_file
from subsetsynth.locks import load_support_lock, load_semantics_lock


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/base.yaml")
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--split",choices=["val","test"],default="val")
    ap.add_argument("--matrix",default=None,help="External 12-config YAML; intended after model freeze.")
    ap.add_argument("--with-lpips",action="store_true")
    ap.add_argument("--unlock-test",default="")
    ap.add_argument("--out",required=True)
    args=ap.parse_args(); cfg=load_config(args.config); load_semantics_lock(cfg); support_lock=load_support_lock(cfg)
    if args.split=="test" and args.unlock_test!=cfg["safety"]["test_unlock_phrase"]:
        raise SystemExit("TEST EMBARGO: provide exact --unlock-test phrase only after model/inference freeze")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SubsetSynthV2(
        img_size=cfg["model"]["img_size"], token_dim=cfg["model"]["token_dim"],
        transformer_layers=cfg["model"]["transformer_layers"], transformer_heads=cfg["model"]["transformer_heads"],
        ff_dim=cfg["model"]["transformer_ff_dim"], physics_enabled=bool(cfg.get("physics",{}).get("enabled",False))).to(device)
    ckpt=load_v2_for_inference(model,args.checkpoint,device,use_ema=True)
    ds=MultiContrastNPYDataset(cfg,args.split); loader=make_eval_loader(ds,cfg)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.matrix:
        rows,cases=evaluate_external_matrix(model,loader,cfg,device,args.matrix,with_lpips=args.with_lpips)
        pd.DataFrame(rows).to_csv(out/f"v2_{args.split}_matrix_slice_metrics.csv",index=False)
        pd.DataFrame(cases).to_csv(out/f"v2_{args.split}_matrix_case_metrics.csv",index=False)
    else:
        rows,cases,summary=evaluate_primitives(model,loader,cfg,device,with_lpips=args.with_lpips,desc=f"v2 {args.split}",support_lock=support_lock)
        save_evaluation(rows,cases,summary,out,f"v2_{args.split}_primitive")
    dump_json({"checkpoint":args.checkpoint,"sha256":sha256_file(args.checkpoint),"split":args.split,"ema":True,"matrix":args.matrix},out/"evaluation_manifest.json")
if __name__=="__main__": main()
