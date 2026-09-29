#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from subsetsynth.config import load_config
from subsetsynth.data import load_split_map, parse_case_id, COHORTS, AVAIL_BY_COHORT
from subsetsynth.locks import load_semantics_lock, expected_mapping_support, SUPPORT_LOCK_VERSION
from subsetsynth.masks import primitive_mappings_for_avail, mapping_name
from subsetsynth.utils import dump_json, sha256_file


def main():
    ap=argparse.ArgumentParser(description="Freeze mapping×cohort support from authoritative splits + materialized files.")
    ap.add_argument("--config",default="configs/base.yaml"); ap.add_argument("--out",default=None)
    args=ap.parse_args(); cfg=load_config(args.config); sem=load_semantics_lock(cfg)
    support=expected_mapping_support()
    canonical=[mapping_name(s,t) for s,t in primitive_mappings_for_avail((1,1,1))]
    case_counts={}; slice_counts={}; cohort_split_cases={}; cohort_split_slices={}
    for cohort in ("A","B","C2","C3"):
        spec=cfg["data"][cohort]; sm=load_split_map(spec["split_csv"])
        cases_by_split=defaultdict(set); slices_by_split=defaultdict(int)
        for fp in Path(spec["dir"]).glob("*.npy"):
            cid=parse_case_id(fp.name); split=sm.get(cid)
            if split is None: continue
            cases_by_split[split].add(cid); slices_by_split[split]+=1
        for split in ("train","val","test"):
            cohort_split_cases[f"{cohort}|{split}"]=len(cases_by_split[split])
            cohort_split_slices[f"{cohort}|{split}"]=slices_by_split[split]
        for src,t in primitive_mappings_for_avail(AVAIL_BY_COHORT[COHORTS[cohort]]):
            m=mapping_name(src,t)
            case_counts[f"{m}|{cohort}"]=len(cases_by_split["val"])
            slice_counts[f"{m}|{cohort}"]=slices_by_split["val"]
    missing=[k for k,v in case_counts.items() if v<=0]
    if missing: raise SystemExit(f"Cannot freeze support: empty validation cells: {missing}")
    lock={
        "version":SUPPORT_LOCK_VERSION,"approved":True,"mapping_order":canonical,"mapping_support":support,
        "validation_case_counts_by_mapping_cohort":case_counts,"validation_slice_counts_by_mapping_cohort":slice_counts,
        "materialized_case_counts_by_cohort_split":cohort_split_cases,"materialized_slice_counts_by_cohort_split":cohort_split_slices,
        "split_sha256":{c:sha256_file(cfg["data"][c]["split_csv"]) for c in ("A","B","C2","C3")},
        "npy_semantics_lock_sha256":sha256_file(cfg["locks"]["npy_semantics_lock"]),
        "policy":"Training and validation aggregation use this identical frozen support set; |C_m| may not adapt to observed results.",
    }
    out=Path(args.out or cfg["locks"]["mapping_cohort_support_lock"]); out.parent.mkdir(parents=True,exist_ok=True); dump_json(lock,out)
    print(json.dumps({"status":"LOCKED","lock":str(out),"mapping_support":support,"validation_cells":case_counts},indent=2))
if __name__=="__main__":main()
