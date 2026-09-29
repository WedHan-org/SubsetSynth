#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from subsetsynth.masks import nonempty_proper_subsets, primitive_mappings_for_avail, SubsetBagManager, mapping_name
from subsetsynth.model import SubsetSynthV2
from subsetsynth.losses import HierarchicalActiveTargetLoss
from subsetsynth.config import load_config
from subsetsynth.data import COHORTS, COHORT_NAMES, AVAIL_BY_COHORT
from subsetsynth.locks import load_support_lock
from subsetsynth.objective import HierarchicalImportanceObjective
from subsetsynth.routing import decode_active_targets


def check(cond, msg):
    if not cond: raise AssertionError(msg)


def build_small_model(cfg, image_size, device):
    return SubsetSynthV2(img_size=image_size,token_dim=32,transformer_layers=1,transformer_heads=4,ff_dim=128,physics_enabled=False).to(device)


def objective_contract(cfg):
    support=load_support_lock(cfg,allow_unlocked=True)
    obj=HierarchicalImportanceObjective.from_config(cfg,support)
    # TEST C: exact normalization identity.
    mass=sum(cell.q*cell.weight for cell in obj.cells.values())
    check(abs(mass-1.0)<1e-12,f"TEST C failed: E_q[a]={mass}")

    # TEST A/B: exact long-run sampler simulation with randomized order but exact cohort counts.
    # 120000 is divisible by 4, 6 and 2, so subset bags complete exact cycles.
    N=120000; schedule=[]
    for cid in range(4): schedule += [cid]*(N//4)
    rng=random.Random(20260821); rng.shuffle(schedule)
    bag=SubsetBagManager(20260821)
    raw=Counter(); weighted=Counter()
    for cid in schedule:
        item=bag.sample_one(cid,AVAIL_BY_COHORT[cid]); c=COHORT_NAMES[cid]
        for t,active in enumerate(item.tgt_mask):
            if not active: continue
            m=mapping_name(item.src_mask,t); raw[(m,c)]+=1; weighted[(m,c)]+=obj.cell(m,c).weight
    for key,cell in obj.cells.items():
        qhat=raw[key]/N; ahat=weighted[key]/N
        check(abs(qhat-cell.q)<1e-12,f"TEST A proposal mismatch {key}: {qhat} vs {cell.q}")
        check(abs(ahat-cell.alpha)<1e-12,f"TEST B weighted target mismatch {key}: {ahat} vs {cell.alpha}")
    mapping_mass=Counter(); cohort_within={}
    for (m,c),v in weighted.items(): mapping_mass[m]+=v/N
    ratio=max(mapping_mass.values())/min(mapping_mass.values())
    check(ratio<1.0000001,f"TEST B mapping target masses unequal: ratio={ratio}")
    return obj


def accumulation_contract(cfg,obj,image_size,device):
    # TEST D: batch=6,accum=1 == micro=2,accum=3 when denominator is fixed EB and no /accum_steps.
    torch.manual_seed(17)
    base=build_small_model(cfg,image_size,device); base.eval()
    m_full=copy.deepcopy(base); m_acc=copy.deepcopy(base); m_full.eval(); m_acc.eval()
    B=6; EB=6
    x=torch.rand(B,3,image_size,image_size,device=device)
    src=torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]],dtype=torch.bool,device=device)
    tgt=~src
    cohort=torch.zeros(B,dtype=torch.long,device=device)

    lf=HierarchicalActiveTargetLoss(cfg,obj).to(device)
    z,sk=m_full.encode(x,src); pr=decode_active_targets(m_full,z,sk,tgt); num,_,_,_,_=lf(pr,x,src,tgt,cohort); (num/EB).backward()
    grads_full={n:p.grad.detach().clone() for n,p in m_full.named_parameters() if p.grad is not None}

    la=HierarchicalActiveTargetLoss(cfg,obj).to(device)
    for start in (0,2,4):
        sl=slice(start,start+2); z,sk=m_acc.encode(x[sl],src[sl]); pr=decode_active_targets(m_acc,z,sk,tgt[sl]); num,_,_,_,_=la(pr,x[sl],src[sl],tgt[sl],cohort[sl]); (num/EB).backward()
    grads_acc={n:p.grad.detach().clone() for n,p in m_acc.named_parameters() if p.grad is not None}
    check(grads_full.keys()==grads_acc.keys(),"TEST D gradient key mismatch")
    worst=0.0
    for n in grads_full:
        torch.testing.assert_close(grads_full[n],grads_acc[n],rtol=2e-4,atol=2e-6,msg=lambda msg,n=n:f"TEST D {n}: {msg}")
        worst=max(worst,float((grads_full[n]-grads_acc[n]).abs().max().cpu()))
    return worst


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/profile.yaml"); ap.add_argument("--device",default="cpu"); ap.add_argument("--image-size",type=int,default=64)
    args=ap.parse_args(); cfg=load_config(args.config); results=[]
    check(len(nonempty_proper_subsets((1,1,1)))==6,"A/B must have 6 source subsets")
    check(len(primitive_mappings_for_avail((1,1,1)))==9,"A/B must have 9 primitive mappings")
    check(len(primitive_mappings_for_avail((1,1,0)))==2,"C2 must have 2 primitive mappings")
    check(len(primitive_mappings_for_avail((1,0,1)))==2,"C3 must have 2 primitive mappings")
    results.append("mask_enumeration")
    bag=SubsetBagManager(123); seen=[bag.sample_one(0,(1,1,1)).src_mask for _ in range(6)]; check(len(set(seen))==6,"subset bag coverage"); results.append("subset_bag_exact_local_coverage")
    root=Path(__file__).resolve().parents[1]; core_text="\n".join(p.read_text(encoding="utf-8") for p in (root/"subsetsynth").glob("*.py")); check("evaluation_configs.yaml" not in core_text,"core must not read evaluation YAML"); check("task_idx" not in (root/"subsetsynth"/"model.py").read_text(),"model must not use task_idx"); results.append("no_predefined_task_id_contract")
    obj=objective_contract(cfg); results += ["TEST_A_proposal_exposure","TEST_B_weighted_target","TEST_C_normalization_identity"]
    device=torch.device(args.device); worst=accumulation_contract(cfg,obj,args.image_size,device); results.append("TEST_D_accumulation_equivalence")
    print(json.dumps({"status":"PASS","checks":results,"expected_mass_per_sample":obj.expected_mass_per_sample,"max_gradient_abs_diff_TEST_D":worst,"objective":obj.to_dict()},indent=2))
if __name__=="__main__":main()
