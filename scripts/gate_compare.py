#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from subsetsynth.config import load_config
from subsetsynth.gate import paired_bootstrap_delta, validate_case_metric_table, assert_same_keyset
from subsetsynth.metrics import mapping_cohort_balanced_summary
from subsetsynth.locks import load_support_lock, load_semantics_lock
from subsetsynth.utils import dump_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/base.yaml")
    ap.add_argument("--v18-cases",required=True)
    ap.add_argument("--v20-cases",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args(); cfg=load_config(args.config); g=cfg["gate"]
    load_semantics_lock(cfg); support_lock=load_support_lock(cfg)
    a=validate_case_metric_table(pd.read_csv(args.v18_cases),require_lpips=True)
    b=validate_case_metric_table(pd.read_csv(args.v20_cases),require_lpips=True)
    assert_same_keyset(a,b)
    mapping=g["primary_mapping"]
    a1=a[a.mapping==mapping].copy(); b1=b[b.mapping==mapping].copy()
    if a1.empty or b1.empty: raise SystemExit(f"Primary mapping absent: {mapping}")
    report={"primary_mapping":mapping,"metrics":{}}
    for metric in ("psnr","ssim","lpips"):
        report["metrics"][metric]=paired_bootstrap_delta(a1,b1,metric,int(g["bootstrap_iterations"]),int(g["bootstrap_seed"]))
    report["criteria"]={
        "psnr_pass":report["metrics"]["psnr"]["ci95_low"]>float(g["psnr_lcb_gt_db"]),
        "ssim_pass":report["metrics"]["ssim"]["ci95_low"]>float(g["ssim_lcb_gt"]),
        "lpips_pass":report["metrics"]["lpips"]["ci95_high"]<float(g["lpips_ucb_lt"]),
    }
    sa=mapping_cohort_balanced_summary(a.to_dict("records"),support_lock=support_lock,strict=True)["overall_mapping_balanced"]["psnr"]
    sb=mapping_cohort_balanced_summary(b.to_dict("records"),support_lock=support_lock,strict=True)["overall_mapping_balanced"]["psnr"]
    delta=float(sb-sa)
    report["whole_model"]={
        "v18_mapping_balanced_psnr":sa,"v20_mapping_balanced_psnr":sb,"delta":delta,
        "guardrail":float(g["mapping_balanced_psnr_guardrail_db"]),
        "pass":delta>=float(g["mapping_balanced_psnr_guardrail_db"]),
    }
    report["GO"] = all(report["criteria"].values()) and report["whole_model"]["pass"]
    dump_json(report,args.out)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
