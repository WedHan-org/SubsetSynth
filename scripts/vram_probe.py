#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from subsetsynth.config import load_config
from subsetsynth.model import SubsetSynthV2
from subsetsynth.losses import HierarchicalActiveTargetLoss, physics_l1_numerator
from subsetsynth.physics import DynamicBlochLayer
from subsetsynth.routing import decode_active_targets
from subsetsynth.locks import load_support_lock
from subsetsynth.objective import HierarchicalImportanceObjective


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/base.yaml"); ap.add_argument("--batches",default="1,2,3,4,6"); args=ap.parse_args()
    cfg=load_config(args.config)
    if not torch.cuda.is_available(): raise SystemExit("CUDA required for VRAM probe")
    support=load_support_lock(cfg,allow_unlocked=True); objective=HierarchicalImportanceObjective.from_config(cfg,support)
    device=torch.device("cuda"); results=[]; H=int(cfg["model"]["img_size"]); phys_on=bool(cfg.get("physics",{}).get("enabled",False))
    for B in [int(x) for x in args.batches.split(',')]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            model=SubsetSynthV2(img_size=H,token_dim=cfg["model"]["token_dim"],transformer_layers=cfg["model"]["transformer_layers"],transformer_heads=cfg["model"]["transformer_heads"],ff_dim=cfg["model"]["transformer_ff_dim"],physics_enabled=phys_on).to(device)
            loss_fn=HierarchicalActiveTargetLoss(cfg,objective).to(device); bloch=DynamicBlochLayer().to(device) if phys_on else None
            # Worst common target count: A/B single source -> two active targets.
            x=torch.rand(B,3,H,H,device=device); src=torch.tensor([[1,0,0]],dtype=torch.bool,device=device).expand(B,-1); tgt=torch.tensor([[0,1,1]],dtype=torch.bool,device=device).expand(B,-1); cohort=torch.zeros(B,dtype=torch.long,device=device)
            torch.cuda.synchronize(); t0=time.time()
            z,skips=model.encode(x,src); preds=decode_active_targets(model,z,skips,tgt); num,_,_,_,_=loss_fn(preds,x,src,tgt,cohort); loss=num/max(B,1)
            if phys_on:
                phys=bloch(model.physics_maps(z),cohort,jitter=True); pnum,_=physics_l1_numerator(phys,x,cohort); loss=loss+float(cfg["physics"]["weight"])*pnum/max(B,1)
            loss.backward(); torch.cuda.synchronize(); elapsed=time.time()-t0
            results.append({"batch":B,"ok":True,"physics_enabled":phys_on,"peak_allocated_gb":torch.cuda.max_memory_allocated()/2**30,"peak_reserved_gb":torch.cuda.max_memory_reserved()/2**30,"forward_backward_seconds":elapsed,"active_decoded_items":2*B})
            del model,loss_fn,x,z,skips,preds,loss
        except torch.cuda.OutOfMemoryError:
            results.append({"batch":B,"ok":False,"oom":True,"physics_enabled":phys_on}); torch.cuda.empty_cache(); break
    print(json.dumps(results,indent=2))
if __name__=="__main__":main()
