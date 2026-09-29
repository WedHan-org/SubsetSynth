#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from subsetsynth.config import load_config
from subsetsynth.data import parse_case_id, load_split_map
from subsetsynth.utils import dump_json, sha256_file


def main():
    ap = argparse.ArgumentParser(description="Phase-0 split/NPY availability audit; no model inference.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="/path/to/local/storage/SubsetSynth_v2_0_phase0_audit")
    ap.add_argument("--hash-test-files", action="store_true", help="Full SHA256 of every materialized test NPY; slower but recommended before clinical cohort lock.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    global_roles = defaultdict(set)
    for cohort in ("A","B","C2","C3"):
        spec = cfg["data"][cohort]
        sm = load_split_map(spec["split_csv"])
        files = defaultdict(list)
        for p in Path(spec["dir"]).glob("*.npy"):
            files[parse_case_id(p.name)].append(p)
        for case_id, split in sm.items():
            global_roles[case_id].add(split)
            n = len(files.get(case_id, []))
            row = {"cohort":cohort,"case_id":case_id,"split":split,"n_npy":n,"available_npy":n>0}
            rows.append(row)
            if n == 0:
                missing.append(row)
        # NPY cases not represented by split CSV are an integrity error.
        extras = sorted(set(files) - set(sm))
        if extras:
            pd.DataFrame({"cohort":cohort,"case_id":extras}).to_csv(out/f"unexpected_npy_cases_{cohort}.csv",index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out/"case_split_npy_availability.csv",index=False)
    pd.DataFrame(missing).to_csv(out/"missing_cases.csv",index=False)
    test = df[df.split=="test"].copy()
    test.to_csv(out/"historical_test_cases_with_npy_status.csv",index=False)
    test[test.available_npy].to_csv(out/"currently_materialized_test_cases.csv",index=False)

    test_file_rows=[]
    for cohort in ("A","B","C2","C3"):
        spec=cfg["data"][cohort]
        sm=load_split_map(spec["split_csv"])
        for fp in sorted(Path(spec["dir"]).glob("*.npy")):
            cid=parse_case_id(fp.name)
            if sm.get(cid)!="test": continue
            st=fp.stat()
            row={"cohort":cohort,"case_id":cid,"file":str(fp),"size":st.st_size,"mtime_ns":st.st_mtime_ns}
            if args.hash_test_files:
                row["sha256"]=sha256_file(fp)
            test_file_rows.append(row)
    pd.DataFrame(test_file_rows).to_csv(out/"test_file_manifest.csv",index=False)

    conflicts = {cid:sorted(roles) for cid,roles in global_roles.items() if len(roles)>1}
    summary = {
        "split_counts": df.groupby(["cohort","split"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "total_counts": df.groupby("split").size().to_dict(),
        "missing_by_split": df[~df.available_npy].groupby("split").size().to_dict(),
        "historical_test_count": int((df.split=="test").sum()),
        "materialized_test_count": int(((df.split=="test") & df.available_npy).sum()),
        "cross_split_case_id_conflicts": conflicts,
        "split_sha256": {k:sha256_file(cfg["data"][k]["split_csv"]) for k in ("A","B","C2","C3")},
        "test_file_hash_mode": "sha256" if args.hash_test_files else "size+mtime_ns",
        "test_npy_file_count": len(test_file_rows),
    }
    dump_json(summary,out/"audit_summary.json")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if conflicts:
        raise SystemExit("FAIL: case ID appears in multiple split roles")

if __name__ == "__main__": main()
