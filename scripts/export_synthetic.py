#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from subsetsynth.config import load_config
from subsetsynth.model import SubsetSynthV2
from subsetsynth.checkpoint import load_v2_for_inference
from subsetsynth.data import MultiContrastNPYDataset, make_eval_loader, COHORT_NAMES
from subsetsynth.masks import MOD_TO_IDX
from subsetsynth.utils import sha256_file, dump_json
from subsetsynth.locks import load_semantics_lock, load_support_lock


def parse_source(s):
    names=[x.strip() for x in s.split('+') if x.strip()]
    m=[0,0,0]
    for n in names: m[MOD_TO_IDX[n]]=1
    return names,tuple(m)


def main():
    ap=argparse.ArgumentParser(description="Export frozen synthetic images for a prespecified mapping.")
    ap.add_argument("--config",default="configs/base.yaml")
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--source",required=True,help="e.g. T1+T2")
    ap.add_argument("--target",required=True,choices=["T1","T2","FS"])
    ap.add_argument("--split",choices=["val","test"],default="test")
    ap.add_argument("--unlock-test",default="")
    ap.add_argument("--out",required=True)
    args=ap.parse_args(); cfg=load_config(args.config); load_semantics_lock(cfg); load_support_lock(cfg)
    if args.split=="test" and args.unlock_test!=cfg["safety"]["test_unlock_phrase"]:
        raise SystemExit("TEST EMBARGO: explicit unlock required")
    src_names,sm=parse_source(args.source); t=MOD_TO_IDX[args.target]
    if sm[t]: raise SystemExit("Target cannot also be a source")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SubsetSynthV2(img_size=cfg["model"]["img_size"],token_dim=cfg["model"]["token_dim"],transformer_layers=cfg["model"]["transformer_layers"],transformer_heads=cfg["model"]["transformer_heads"],ff_dim=cfg["model"]["transformer_ff_dim"], physics_enabled=bool(cfg.get("physics",{}).get("enabled",False))).to(device)
    load_v2_for_inference(model,args.checkpoint,device,use_ema=True)
    ds=MultiContrastNPYDataset(cfg,args.split); loader=make_eval_loader(ds,cfg)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with torch.no_grad():
        for batch in loader:
            images=batch["images"].to(device).float(); avail=batch["avail_mask"].to(device).bool()
            needed=torch.tensor(sm,dtype=torch.bool,device=device)
            eligible=(avail | ~needed).all(dim=1) & avail[:,t]
            idx=torch.where(eligible)[0]
            if idx.numel()==0: continue
            x=images[idx]; src=needed.view(1,3).expand(idx.numel(),-1)
            z,skips=model.encode(x,src)
            tid=torch.full((idx.numel(),),t,device=device,dtype=torch.long)
            pred=model.decode(z,skips,tid).cpu().numpy()
            for j,orig in enumerate(idx.cpu().tolist()):
                case=batch["case_id"][orig]; cohort=COHORT_NAMES[int(batch["cohort_id"][orig])]
                base=Path(batch["path"][orig]).stem
                d=out/cohort/case; d.mkdir(parents=True,exist_ok=True)
                fp=d/f"{base}__{'-'.join(src_names)}_to_{args.target}.npy"
                np.save(fp,pred[j,0].astype(np.float32))
                manifest.append({"case_id":case,"cohort":cohort,"source":"+".join(src_names),"target":args.target,"source_npy":batch["path"][orig],"synthetic_npy":str(fp)})
    pd.DataFrame(manifest).to_csv(out/"synthetic_manifest.csv",index=False)
    dump_json({"checkpoint":args.checkpoint,"checkpoint_sha256":sha256_file(args.checkpoint),"ema":True,"split":args.split,"source":src_names,"target":args.target,"n_slices":len(manifest)},out/"export_manifest.json")
    print(f"exported {len(manifest)} slices -> {out}")
if __name__=="__main__":main()
