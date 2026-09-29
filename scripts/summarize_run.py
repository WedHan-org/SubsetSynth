#!/usr/bin/env python3
from pathlib import Path
import argparse,json
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run_dir"); args=ap.parse_args()
    r=Path(args.run_dir)
    lines=[f"# Run Summary — {r.name}",""]
    man=r/"RUN_MANIFEST.json"
    if man.exists():
        m=json.loads(man.read_text()); lines += ["## Manifest","", "```json", json.dumps(m,indent=2,ensure_ascii=False), "```",""]
    tr=r/"records"/"train_epoch.csv"
    if tr.exists():
        df=pd.read_csv(tr); last=df.iloc[-1].to_dict(); lines += ["## Training","",f"- Epochs completed: {int(last['epoch'])}",f"- Last epoch seconds: {last.get('seconds','NA')}",f"- Last reconstruction objective estimate: {last.get('objective_recon_estimate','NA')}",""]
    va=r/"records"/"validation_events.csv"
    if va.exists():
        df=pd.read_csv(va); best=df.sort_values(["mapping_balanced_psnr","mapping_balanced_ssim"],ascending=False).iloc[0]
        lines += ["## Validation","",f"- Best logged mapping-balanced PSNR: {best.mapping_balanced_psnr:.4f} dB at epoch {int(best.epoch)}",f"- SSIM at that event: {best.mapping_balanced_ssim:.6f}",""]
    out=r/"RUN_SUMMARY.md"; out.write_text("\n".join(lines),encoding="utf-8"); print(out)
if __name__=="__main__":main()
