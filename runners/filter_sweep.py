"""Compare filtering scores and sampling-step windows."""
from __future__ import annotations
import argparse, csv, itertools, json, time
from pathlib import Path

import numpy as np
import torch

from dino_uncertainty.metrics.fid import fid_from_features, reference_stats
from dino_uncertainty.config import FILTER_SCORES, PROTOCOL

POINTS = (0.30, 0.40, 0.50, 0.60, 0.80, 0.90)

def load_pool_scores(root: Path, n: int):
    """Load IDs, labels, Inception features and filtering scores without loading images."""
    ids, labels, feats, scores = [], [], [], {}
    for f in sorted((root / "pool" / "shards").glob("*.pt")):
        r = torch.load(f, map_location="cpu")
        ids.append(torch.as_tensor(r["ids"])); labels.append(r["labels"]); feats.append(r["incep"])
        for k, v in (r.get("scores") or {}).items():
            scores.setdefault(float(k), []).append(torch.as_tensor(v))
    if not ids:
        raise SystemExit(f"no pool shards under {root}")
    ids = torch.cat(ids); o = torch.argsort(ids)
    out = {"ids": ids[o], "labels": torch.cat(labels)[o], "incep": torch.cat(feats)[o],
           "scores": {k: torch.cat(v)[o] for k, v in scores.items()}}
    if len(out["ids"]) < n:
        raise SystemExit(f"pool has {len(out['ids'])}, need {n}")
    if not torch.equal(out["ids"][:n], torch.arange(n)):
        raise SystemExit(f"pool must contain unique image IDs 0..{n - 1}")
    return out

def load_dense_scores(d: Path, agg: str):
    """Load saved scores for each sampling step."""
    fs = sorted((d / "shards").glob("*.pt"))
    if not fs:
        return None
    ids, per_step, fracs = [], {}, None
    for f in fs:
        r = torch.load(f, map_location="cpu")
        ids.append(torch.as_tensor(r["ids"])); fracs = r["fractions"]
        for i, v in r[agg].items():
            per_step.setdefault(int(i), []).append(torch.as_tensor(v))
    ids = torch.cat(ids); o = torch.argsort(ids)
    return {"ids": ids[o], "fractions": fracs,
            "per_step": {i: torch.cat(v)[o] for i, v in per_step.items()}}

def range_candidates(per_step: dict, n_steps: int, width: int | None = None):
    """Build consecutive step windows and their average scores."""
    steps = sorted(per_step)
    if not steps:
        return []
    if width == 1:
        return [(f"step{i}", "exploratory", per_step[i]) for i in steps]
    runs, cur = [], [steps[0]]
    for a, b in zip(steps, steps[1:]):
        (cur.append(b) if b == a + 1 else (runs.append(cur), cur := [b]))
    runs.append(cur)
    band = max(runs, key=len)  # Longest group of consecutive scored steps.
    out = ([] if width is not None else
           [(f"step{i}", "exploratory", per_step[i]) for i in steps])
    for a in range(len(band)):
        for b in range(a + 1, len(band)):
            lo, hi = band[a], band[b]
            if width is not None and hi - lo + 1 != width:
                continue
            inside = list(range(lo, hi + 1))
            assert all(i in per_step for i in inside), (lo, hi)
            name = f"win{lo}_{hi}"                  # frac {lo/n_steps:.2f}-{(hi+1)/n_steps:.2f}
            out.append((name, "exploratory",
                        sum(per_step[i] for i in inside) / len(inside)))
    return out

def load_replay_scores(d: Path):
    """Load scores from an optional comparison run."""
    fs = sorted((d / "shards").glob("*.pt"))
    if not fs:
        return None
    ids, scores = [], {}
    for f in fs:
        r = torch.load(f, map_location="cpu")
        ids.append(torch.as_tensor(r["ids"]))
        for k, v in r["scores"].items():
            scores.setdefault(float(k), []).append(torch.as_tensor(v))
    ids = torch.cat(ids); o = torch.argsort(ids)
    return {"ids": ids[o], "scores": {k: torch.cat(v)[o] for k, v in scores.items()}}

def candidates(S: dict, zs: dict):
    """Return the score choices to compare. Lower scores are kept."""
    out = []
    for p in POINTS:
        out.append((f"single_p{p:g}", "exploratory", S[p]))
    for a, b in itertools.combinations(POINTS, 2):
        name = f"pair_p{a:g}_p{b:g}"
        kind = "confirmatory" if (a, b) in FILTER_SCORES.values() else "exploratory"
        out.append((name, kind, (S[a] + S[b]) / 2))
    out.append(("all6_mean", "exploratory", sum(S[p] for p in POINTS) / len(POINTS)))
    out.append(("all6_zmean", "exploratory", sum(zs[p] for p in POINTS) / len(POINTS)))
    out.append(("all6_zmax", "exploratory", torch.stack([zs[p] for p in POINTS]).max(0).values))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[128, 256, 512])
    ap.add_argument("--pool-n", type=int, default=12000)
    ap.add_argument("--keep", type=int, default=10000)
    ap.add_argument("--random-subsets", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--root", default="outputs/final_experiments")
    ap.add_argument("--score-source", default="pool",
                    help="pool uses scores saved during generation; a folder uses its saved scores")
    ap.add_argument("--tag", default=None, help="output name; defaults to the score source")
    ap.add_argument("--dense", default=None,
                    help="folder of per-step scores used to compare consecutive windows")
    ap.add_argument("--dense-agg", default="mae_top10",
                    help="score to use: mae_top10, mae_all, dino_top10 or dino_all")
    ap.add_argument("--range-width", type=int, default=None,
                    help="only compare windows with this many steps")
    ap.add_argument("--skip-sparse", action="store_true",
                    help="skip the six fixed single-step and pair scores")
    a = ap.parse_args()
    if not 2 <= a.keep <= a.pool_n or a.random_subsets < 1:
        ap.error("keep must be between 2 and pool-n; random-subsets must be positive")
    if a.range_width is not None and a.range_width < 1:
        ap.error("range-width must be positive")

    res = a.resolution; ds = PROTOCOL[res]["dataset"]
    root = Path(a.root) / f"imagenet{res}"
    tag = a.tag or ("top10_selected" if a.score_source == "pool" else Path(a.score_source).name)
    out = root / "analysis_sweep"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pool = load_pool_scores(root, a.pool_n)
    ids, labels = pool["ids"][:a.pool_n], pool["labels"][:a.pool_n]
    feats = pool["incep"][:a.pool_n].numpy()
    if a.score_source == "pool":
        S = {k: v[:a.pool_n] for k, v in pool["scores"].items()}
        aggregation = "mean raw variance over the top-10% selected locations (spec)"
    else:
        rep = load_replay_scores(Path(a.score_source) if "/" in a.score_source
                                 else root / a.score_source)
        if rep is None:
            raise SystemExit(f"no saved scores at {a.score_source}; create them first or "
                             "use --score-source pool")
        if not torch.equal(rep["ids"][:a.pool_n], ids):
            raise SystemExit("replay IDs do not match the pool")
        S = {k: v[:a.pool_n] for k, v in rep["scores"].items()}
        aggregation = f"replay scores from {a.score_source}"
    missing = [p for p in POINTS if p not in S]
    if missing:
        raise SystemExit(f"missing score points {missing}")
    zs = {p: (S[p] - S[p].mean()) / S[p].std().clamp_min(1e-12) for p in POINTS}

    extra = []
    if a.dense:
        dd = Path(a.dense) if "/" in a.dense else root / a.dense
        dense = load_dense_scores(dd, a.dense_agg)
        if dense is None:
            raise SystemExit(f"no dense shards at {dd}")
        if not torch.equal(dense["ids"][:a.pool_n], ids):
            raise SystemExit("dense replay IDs do not match the pool")
        per_step = {i: v[:a.pool_n] for i, v in dense["per_step"].items()}
        extra = range_candidates(per_step, PROTOCOL[res]["steps"], a.range_width)
        tag = f"{tag}_dense_{a.dense_agg}"
        aggregation = f"dense per-step, aggregation={a.dense_agg}, from {dd}"
        print(f"[sweep] dense: {len(per_step)} steps -> {len(extra)} range candidates", flush=True)

    base = [] if a.skip_sparse else candidates(S, zs)
    if not base and not extra:
        ap.error("no score windows selected; check dense scores and range-width")
    _, _, prov = reference_stats(ds)
    print(f"[sweep] {ds} n={len(ids)} keep={a.keep} source={tag}", flush=True)
    fid_pool, _ = fid_from_features(feats, ds)
    per_fid = time.time() - t0
    print(f"[sweep] full-pool FID ({len(ids)}) = {fid_pool:.4f}   [{per_fid:.0f}s per FID]", flush=True)

    rng = np.random.default_rng(a.seed)
    rand = []
    for k in range(a.random_subsets):
        sel = rng.choice(len(ids), size=a.keep, replace=False)
        f, _ = fid_from_features(feats[sel], ds); rand.append(f)
        print(f"[sweep]   random {k+1}/{a.random_subsets}: {f:.4f}", flush=True)
    r_mean = float(np.mean(rand))
    r_std = float(np.std(rand, ddof=1)) if len(rand) > 1 else 0.0
    print(f"[sweep] random retention = {r_mean:.4f} +/- {r_std:.4f}", flush=True)

    rows = []
    for name, kind, score in base + extra:
        keep = np.argsort(score.numpy(), kind="stable")[:a.keep]
        f, _ = fid_from_features(feats[keep], ds)
        cls, cnt = np.unique(labels.numpy()[keep], return_counts=True)
        rows.append({"score": name, "kind": kind, "fid_retained": f,
                     "delta_vs_random": f - r_mean,
                     "z_vs_random": (f - r_mean) / r_std if r_std > 0 else None,
                     "delta_vs_pool": f - fid_pool,
                     "n_classes": int(len(cls)), "class_count_std": float(cnt.std())})
        print(f"[sweep] {kind:13s} {name:22s} FID {f:8.4f}  vs random {f-r_mean:+7.4f}", flush=True)

    rows.sort(key=lambda r: r["fid_retained"])
    with open(out / f"sweep_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    conf = [r for r in rows if r["kind"] == "confirmatory"]
    summary = {
        "dataset": ds, "score_source": tag, "aggregation": aggregation,
        "pool_n": int(len(ids)), "keep": a.keep,
        "fid_pool_full": fid_pool,
        "fid_random_mean": r_mean, "fid_random_std": r_std,
        "random_subsets": a.random_subsets, "random_seed": a.seed,
        "confirmatory": conf,
        "best_overall": rows[0],
        "all_rows": rows,
        "fid_reference": prov,
        "seconds_per_fid": per_fid,
        "caveat": "The confirmatory rows were fixed before any FID was computed. The "
                  "exploratory rows were not: choosing the lowest of them and reporting it as "
                  "the method is selection on the evaluation set. Use the spread to show how "
                  "much the choice matters, and quote the prespecified middle score as primary.",
        "runtime_sec": time.time() - t0,
    }
    (out / f"sweep_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[sweep] best overall: {rows[0]['score']} FID {rows[0]['fid_retained']:.4f}", flush=True)
    for r in conf:
        print(f"[sweep] prespecified: {r['score']:22s} FID {r['fid_retained']:.4f}", flush=True)
    print(f"[sweep] wrote {out}/sweep_{tag}.{{csv,json}}  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
