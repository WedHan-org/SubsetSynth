#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from subsetsynth.config import load_config
from subsetsynth.semantics import orientation_contract
from subsetsynth.utils import dump_json, sha256_file
from subsetsynth.locks import SEMANTIC_LOCK_VERSION

PHRASE="I_HAVE_VISUALLY_VERIFIED_NPY_MODALITY_SEMANTICS_AND_ORIENTATION"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/base.yaml")
    ap.add_argument("--audit-draft",required=True)
    ap.add_argument("--confirm",required=True)
    ap.add_argument("--note",default="Visual review completed against expected modality labels and model-canonical orientation.")
    ap.add_argument("--out",default=None)
    args=ap.parse_args(); cfg=load_config(args.config)
    if args.confirm != PHRASE: raise SystemExit(f"Refusing approval. Exact --confirm phrase required: {PHRASE}")
    draft_path=Path(args.audit_draft); draft=json.loads(draft_path.read_text(encoding="utf-8"))
    if draft.get("version")!=SEMANTIC_LOCK_VERSION: raise SystemExit("Unsupported audit draft version")
    if draft.get("schema")!=cfg["data_schema"]: raise SystemExit("Audit draft schema differs from config")
    expected_orientation=orientation_contract()
    if draft.get("orientation_contract")!=expected_orientation:
        raise SystemExit("Audit draft orientation contract differs from current shared code contract")
    current_hashes={c:sha256_file(cfg["data"][c]["split_csv"]) for c in ("A","B","C2","C3")}
    if draft.get("split_sha256")!=current_hashes: raise SystemExit("Split hashes changed since semantic audit")
    lock={
        "version":SEMANTIC_LOCK_VERSION,
        "approved":True,
        "schema":cfg["data_schema"],
        "orientation_contract":expected_orientation,
        "split_sha256":current_hashes,
        "audit_draft":str(draft_path),
        "audit_draft_sha256":sha256_file(draft_path),
        "human_confirmation_phrase":PHRASE,
        "approval_note":args.note,
        "sample_evidence":draft.get("samples",[]),
    }
    out=Path(args.out or cfg["locks"]["npy_semantics_lock"]); out.parent.mkdir(parents=True,exist_ok=True); dump_json(lock,out)
    print(json.dumps({"status":"APPROVED","lock":str(out),"sha256":sha256_file(out)},indent=2))
if __name__=="__main__":main()
