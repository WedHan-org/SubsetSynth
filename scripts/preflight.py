#!/usr/bin/env python3
from pathlib import Path
import argparse, importlib.util, json, shutil, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from subsetsynth.config import load_config
from subsetsynth.locks import load_semantics_lock, load_support_lock


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/base.yaml"); ap.add_argument("--allow-missing-locks",action="store_true"); args=ap.parse_args(); cfg=load_config(args.config)
    paths={}; ok=True
    for c in ("A","B","C2","C3"):
        for key in ("dir","split_csv"):
            p=Path(cfg["data"][c][key]); exists=p.exists(); paths[f"{c}.{key}"]={"path":str(p),"exists":exists}; ok &= exists
    locks={}; locks_ok=True
    for name,key,fn in (("npy_semantics","npy_semantics_lock",load_semantics_lock),("mapping_cohort_support","mapping_cohort_support_lock",load_support_lock)):
        p=Path(cfg["locks"][key]); status={"path":str(p),"exists":p.exists(),"valid":False,"error":None}
        if p.exists():
            try: fn(cfg); status["valid"]=True
            except Exception as e: status["error"]=f"{type(e).__name__}: {e}"; locks_ok=False
        else: locks_ok=False
        locks[name]=status
    out_root=Path(cfg["paths"]["output_root"]); probe=out_root if out_root.exists() else out_root.parent
    while not probe.exists() and probe.parent!=probe: probe=probe.parent
    du=shutil.disk_usage(probe)
    report={
        "paths_ok":bool(ok),"locks_ok":bool(locks_ok),"paths":paths,"locks":locks,
        "python":sys.version,"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"free_disk_gb":du.free/2**30,
        "lpips_installed":importlib.util.find_spec("lpips") is not None,"pyyaml_installed":importlib.util.find_spec("yaml") is not None,"opencv_installed":importlib.util.find_spec("cv2") is not None,
        "physics_enabled":bool(cfg.get("physics",{}).get("enabled",False)),
    }
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not ok: raise SystemExit(2)
    if not locks_ok and not args.allow_missing_locks: raise SystemExit(3)
if __name__=="__main__":main()
