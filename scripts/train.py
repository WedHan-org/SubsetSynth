#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from subsetsynth.config import load_config
from subsetsynth.model import SubsetSynthV2
from subsetsynth.engine import train
from subsetsynth.utils import seed_everything


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/train.yaml")
    ap.add_argument("--resume",default=None)
    args=ap.parse_args(); cfg=load_config(args.config)
    seed_everything(int(cfg["reproducibility"]["seed"]),bool(cfg["reproducibility"].get("deterministic",False)))
    model=SubsetSynthV2(
        img_size=cfg["model"]["img_size"], token_dim=cfg["model"]["token_dim"],
        transformer_layers=cfg["model"]["transformer_layers"], transformer_heads=cfg["model"]["transformer_heads"],
        ff_dim=cfg["model"]["transformer_ff_dim"], physics_enabled=bool(cfg.get("physics",{}).get("enabled",False)))
    run_dir=train(cfg,model,Path(__file__).resolve().parents[1],mode="train",resume=args.resume)
    print(f"TRAIN_RUN={run_dir}")
if __name__=="__main__": main()
