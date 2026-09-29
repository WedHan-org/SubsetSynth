from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_KEYS = ["case_id","cohort","mapping"]


def validate_case_metric_table(df: pd.DataFrame, require_lpips: bool = False) -> pd.DataFrame:
    required = set(REQUIRED_KEYS + ["psnr","ssim","mae"] + (["lpips"] if require_lpips else []))
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Case metric table missing columns: {sorted(missing)}")
    out=df.copy()
    for k in REQUIRED_KEYS:
        out[k]=out[k].astype(str)
    if out.duplicated(REQUIRED_KEYS).any():
        dups=out.loc[out.duplicated(REQUIRED_KEYS,keep=False),REQUIRED_KEYS].head(20).to_dict("records")
        raise ValueError(f"Duplicate case/cohort/mapping rows: {dups}")
    metrics=["psnr","ssim","mae"] + (["lpips"] if require_lpips else [])
    for m in metrics:
        vals=pd.to_numeric(out[m],errors="coerce")
        if (~np.isfinite(vals.to_numpy())).any():
            raise ValueError(f"Non-finite/non-numeric metric {m}")
        out[m]=vals
    return out


def assert_same_keyset(a: pd.DataFrame, b: pd.DataFrame) -> None:
    ka=set(map(tuple,a[REQUIRED_KEYS].to_numpy()))
    kb=set(map(tuple,b[REQUIRED_KEYS].to_numpy()))
    if ka != kb:
        only_a=sorted(ka-kb)[:20]; only_b=sorted(kb-ka)[:20]
        raise ValueError(f"Unpaired case metric keysets: only_a={only_a}, only_b={only_b}")


def paired_bootstrap_delta(a, b, metric, n_boot=10000, seed=20260821):
    a=validate_case_metric_table(a,require_lpips=(metric=="lpips"))
    b=validate_case_metric_table(b,require_lpips=(metric=="lpips"))
    assert_same_keyset(a,b)
    merged = a[REQUIRED_KEYS+[metric]].merge(
        b[REQUIRED_KEYS+[metric]], on=REQUIRED_KEYS, suffixes=("_v18","_v20"), how="inner", validate="one_to_one"
    )
    if len(merged) < 2:
        raise RuntimeError(f"Too few paired case×mapping rows: {len(merged)}")
    delta = merged[f"{metric}_v20"].to_numpy() - merged[f"{metric}_v18"].to_numpy()
    rng = np.random.default_rng(seed)
    n = len(delta)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0,n,size=n)
        boots[i] = delta[idx].mean()
    return {
        "n_rows": n,
        "n_cases": int(merged["case_id"].nunique()),
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.percentile(boots,2.5)),
        "ci95_high": float(np.percentile(boots,97.5)),
    }
