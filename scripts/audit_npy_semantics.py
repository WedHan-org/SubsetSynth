#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from subsetsynth.config import load_config
from subsetsynth.data import load_split_map, parse_case_id, normalize_to_01
from subsetsynth.semantics import canonicalize_orientation, orientation_contract
from subsetsynth.utils import dump_json, sha256_file
from subsetsynth.locks import SEMANTIC_LOCK_VERSION


def chw(a):
    a=np.asarray(a)
    if a.ndim==2: return a[None]
    if a.ndim==3 and a.shape[-1] <= 3: return a.transpose(2,0,1)
    return a


def tile(images, labels, width=320):
    panels=[]
    for im,label in zip(images,labels):
        x=normalize_to_01(np.asarray(im,float))
        x=cv2.resize(x,(width,width),interpolation=cv2.INTER_LINEAR)
        x=(np.clip(x,0,1)*255).astype(np.uint8)
        canvas=cv2.cvtColor(x,cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas,label,(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(canvas,label,(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1,cv2.LINE_AA)
        panels.append(canvas)
    return np.concatenate(panels,axis=1)


def main():
    ap=argparse.ArgumentParser(
        description=(
            "Generate structural + visual NPY modality/orientation semantic audit. "
            "Both raw NPY and model-canonical contact sheets are exported; no model inference."
        )
    )
    ap.add_argument("--config",default="configs/base.yaml")
    ap.add_argument("--out",default="/path/to/local/storage/SubsetSynth_v2_0b_semantics_audit_v2_orientation")
    ap.add_argument("--samples-per-cohort",type=int,default=6)
    ap.add_argument("--seed",type=int,default=20260821)
    args=ap.parse_args(); cfg=load_config(args.config)
    out=Path(args.out)
    (out/"contact_sheets").mkdir(parents=True,exist_ok=True)
    (out/"raw_contact_sheets").mkdir(parents=True,exist_ok=True)
    rng=random.Random(args.seed)
    orient=orientation_contract()
    report={
        "version":SEMANTIC_LOCK_VERSION,
        "approved":False,
        "schema":cfg["data_schema"],
        "orientation_contract":orient,
        "split_sha256":{},
        "samples":[],
    }

    for cohort in ("A","B","C2","C3"):
        spec=cfg["data"][cohort]
        report["split_sha256"][cohort]=sha256_file(spec["split_csv"])
        sm=load_split_map(spec["split_csv"])
        by_case={}
        for fp in sorted(Path(spec["dir"]).glob("*.npy")):
            cid=parse_case_id(fp.name)
            if cid in sm and cid not in by_case:
                by_case[cid]=fp
        case_ids=sorted(by_case)
        if not case_ids: raise SystemExit(f"No auditable NPY cases for {cohort}")
        chosen=rng.sample(case_ids,min(args.samples_per_cohort,len(case_ids)))
        expected=cfg["data_schema"][cohort]
        for cid in chosen:
            fp=by_case[cid]
            obj=np.load(fp,allow_pickle=True).item()
            if set(("input","label"))-set(obj): raise SystemExit(f"Missing input/label keys: {fp}")
            inp_raw,lab_raw=chw(obj["input"]),chw(obj["label"])
            exp_in=list(expected["input_order"]); exp_lab=list(expected["label_order"])
            if inp_raw.shape[0]!=len(exp_in) or lab_raw.shape[0]!=len(exp_lab):
                raise SystemExit(f"Shape/schema mismatch {fp}: input={inp_raw.shape}, label={lab_raw.shape}, expected={expected}")

            labels=[f"input[{i}] expected {m}" for i,m in enumerate(exp_in)] + [f"label[{i}] expected {m}" for i,m in enumerate(exp_lab)]

            raw_images=[inp_raw[i] for i in range(inp_raw.shape[0])] + [lab_raw[i] for i in range(lab_raw.shape[0])]
            raw_sheet=tile(raw_images,labels)
            raw_sheet_path=out/"raw_contact_sheets"/f"{cohort}__{cid}.png"
            cv2.imwrite(str(raw_sheet_path),raw_sheet)

            # CRITICAL: use exactly the same shared orientation function as the model loader.
            inp=canonicalize_orientation(inp_raw,cohort)
            lab=canonicalize_orientation(lab_raw,cohort)
            images=[inp[i] for i in range(inp.shape[0])] + [lab[i] for i in range(lab.shape[0])]
            sheet=tile(images,labels)
            sheet_path=out/"contact_sheets"/f"{cohort}__{cid}.png"
            cv2.imwrite(str(sheet_path),sheet)

            report["samples"].append({
                "cohort":cohort,
                "case_id":cid,
                "split":sm[cid],
                "npy":str(fp),
                "npy_sha256":sha256_file(fp),
                "input_shape":list(inp_raw.shape),
                "label_shape":list(lab_raw.shape),
                "raw_contact_sheet":str(raw_sheet_path),
                "contact_sheet":str(sheet_path),
                "expected_input_order":exp_in,
                "expected_label_order":exp_lab,
                "orientation_transform":orient[cohort],
            })
    draft=out/"SEMANTICS_AUDIT_DRAFT.json"
    dump_json(report,draft)
    print(json.dumps({
        "status":"REQUIRES_HUMAN_VISUAL_REVIEW_OF_MODALITY_AND_CANONICAL_ORIENTATION",
        "draft":str(draft),
        "canonical_contact_sheets":str(out/'contact_sheets'),
        "raw_contact_sheets":str(out/'raw_contact_sheets'),
        "n_samples":len(report['samples']),
    },indent=2))

if __name__=="__main__": main()
