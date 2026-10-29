"""Measure FID after keeping different numbers of samples."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch

from dino_uncertainty.metrics.fid import fid_from_features
from dino_uncertainty.config import PROTOCOL
from runners.filter_sweep import load_pool_scores, load_dense_scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[128, 256, 512])
    ap.add_argument("--dense", required=True, help="dense score directory")
    ap.add_argument("--dense-agg", default="mae_top10")
    ap.add_argument("--window", required=True, help="'lo-hi' inclusive step window to score by")
    ap.add_argument("--keeps", type=int, nargs="+",
                    default=[6000, 7000, 8000, 9000, 10000, 11000, 12000])
    ap.add_argument("--pool-n", type=int, default=12000)
    ap.add_argument("--random-subsets", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--root", default="outputs/final_experiments")
    a = ap.parse_args()
    if a.pool_n < 2 or a.random_subsets < 1 or any(k < 2 for k in a.keeps):
        ap.error("pool-n and keeps must be at least 2; random-subsets must be positive")
    if not any(k <= a.pool_n for k in a.keeps):
        ap.error("at least one keep value must be no larger than pool-n")

    ds = PROTOCOL[a.resolution]["dataset"]
    root = Path(a.root) / f"imagenet{a.resolution}"
    pool = load_pool_scores(root, a.pool_n)
    feats = pool["incep"][:a.pool_n].numpy()
    dense_path = Path(a.dense) if "/" in a.dense else root / a.dense
    dense = load_dense_scores(dense_path, a.dense_agg)
    if dense is None:
        raise SystemExit(f"no dense shards at {a.dense}")
    if not torch.equal(dense["ids"][:a.pool_n], pool["ids"][:a.pool_n]):
        raise SystemExit("dense score IDs do not match the pool")
    lo, hi = (int(x) for x in a.window.split("-"))
    if not 0 <= lo <= hi < PROTOCOL[a.resolution]["steps"]:
        ap.error("window must be an inclusive range inside the sampling steps")
    per = dense["per_step"]
    missing = [i for i in range(lo, hi + 1) if i not in per]
    if missing:
        raise SystemExit(f"dense scores missing steps {missing}")
    score = sum(per[i][:a.pool_n] for i in range(lo, hi + 1)) / (hi - lo + 1)
    order = np.argsort(score.numpy(), kind="stable")

    rng = np.random.default_rng(a.seed)
    rows, t0 = [], time.time()
    for k in a.keeps:
        if k > a.pool_n:
            continue
        f_sel, _ = fid_from_features(feats[order[:k]], ds)
        rand = [fid_from_features(feats[rng.choice(a.pool_n, size=k, replace=False)], ds)[0]
                for _ in range(a.random_subsets)] if k < a.pool_n else [f_sel]
        m, s = float(np.mean(rand)), float(np.std(rand, ddof=1)) if len(rand) > 1 else 0.0
        rows.append({"keep": k, "fid_selected": f_sel, "fid_random_mean": m,
                     "fid_random_std": s, "delta_vs_random": f_sel - m,
                     "z_vs_random": (f_sel - m) / s if s > 0 else None})
        print(f"[retention] keep {k:6d}  selected {f_sel:8.4f}  random {m:8.4f}+/-{s:.4f}  "
              f"delta {f_sel - m:+8.4f}", flush=True)

    out = root / "analysis_sweep"; out.mkdir(parents=True, exist_ok=True)
    tag = f"retention_{a.dense_agg}_w{lo}_{hi}"
    with open(out / f"{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / f"{tag}.json").write_text(json.dumps(
        {"dataset": ds, "window": [lo, hi], "aggregation": a.dense_agg, "pool_n": a.pool_n,
         "random_subsets": a.random_subsets, "random_seed": a.seed, "rows": rows,
         "claim": "the comparison is selected vs RANDOM at the same k; FID falling with k is a "
                  "sample-size artefact and is not evidence of filtering",
         "runtime_sec": time.time() - t0}, indent=2) + "\n")
    print(f"[retention] wrote {out / tag}.csv/.json", flush=True)

if __name__ == "__main__":
    main()
