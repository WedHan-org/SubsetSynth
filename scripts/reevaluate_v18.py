#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from subsetsynth.config import load_config
from subsetsynth.data import MultiContrastNPYDataset, make_eval_loader
from subsetsynth.legacy import build_legacy_model, evaluate_legacy_primitives
from subsetsynth.evaluator import save_evaluation
from subsetsynth.utils import dump_json, sha256_file
from subsetsynth.locks import load_support_lock, load_semantics_lock


def main():
    ap=argparse.ArgumentParser(description="Re-evaluate v1.8.2 using the exact same v2 common metric implementation.")
    ap.add_argument("--config",default="configs/base.yaml")
    ap.add_argument("--legacy-script",required=True)
    ap.add_argument("--checkpoint",required=True,help="Explicit path required; package does not guess latest/best.")
    ap.add_argument("--split",choices=["val","test"],default="val")
    ap.add_argument("--with-lpips",action="store_true")
    ap.add_argument("--unlock-test",default="")
    ap.add_argument("--out",required=True)
    args=ap.parse_args(); cfg=load_config(args.config); load_semantics_lock(cfg); support_lock=load_support_lock(cfg)
    if args.split=="test" and args.unlock_test!=cfg["safety"]["test_unlock_phrase"]:
        raise SystemExit("TEST EMBARGO: explicit unlock required")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,ckpt=build_legacy_model(args.legacy_script,args.checkpoint,device,use_ema=True)
    ds=MultiContrastNPYDataset(cfg,args.split); loader=make_eval_loader(ds,cfg)
    rows,cases,summary=evaluate_legacy_primitives(model,loader,cfg,device,with_lpips=args.with_lpips,support_lock=support_lock)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    save_evaluation(rows,cases,summary,out,f"v18_{args.split}_primitive")
    dump_json({"legacy_script":args.legacy_script,"checkpoint":args.checkpoint,"sha256":sha256_file(args.checkpoint),"ema":True,"split":args.split},out/"evaluation_manifest.json")
if __name__=="__main__": main()
