"""Calculate filtering scores at many sampling steps."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import torch
import torch.nn.functional as F

from dino_uncertainty.config import (FINAL, PROTOCOL, DENSE_LO, DENSE_HI, dense_indices,
                          filter_indices)
from dino_uncertainty.guidance.core import (masked_values, probe_noise, to_latent,
                                            build_mask_latent)
from runners.generate import DSched, SCORE_STREAM, atomic_save, atomic_json, load_backend
from dino_uncertainty.provenance import source_manifest, write_or_verify_manifest, write_run_settings
from dino_uncertainty.paths import PROJECT_ROOT


def validate_shard(path, lo, hi, labels, fractions):
    """Reject incomplete or incompatible saved scores before reusing them."""
    record = torch.load(path, map_location="cpu")
    if not isinstance(record, dict):
        raise RuntimeError(f"invalid score record in {path}")
    ids, saved_labels = record.get("ids"), record.get("labels")
    if ids is None or torch.as_tensor(ids).tolist() != list(range(lo, hi)):
        raise RuntimeError(f"invalid image IDs in {path}")
    if saved_labels is None or not torch.equal(torch.as_tensor(saved_labels), labels[lo:hi]):
        raise RuntimeError(f"invalid class labels in {path}")
    if record.get("fractions") != fractions:
        raise RuntimeError(f"incompatible scoring steps in {path}")
    for name in ("mae_top10", "mae_all", "dino_top10", "dino_all"):
        per_step = record.get(name)
        if not isinstance(per_step, dict) or set(per_step) != set(fractions):
            raise RuntimeError(f"missing or unexpected {name} steps in {path}")
        for values in per_step.values():
            if (not isinstance(values, torch.Tensor) or values.shape != (hi - lo,)
                    or not values.is_floating_point() or not torch.isfinite(values).all()
                    or (values < 0).any()):
                raise RuntimeError(f"invalid {name} scores in {path}")

@torch.no_grad()
def probe_images(bundle, z, y, t, A, sig, ids, M, mode, radius):
    """Return the predicted images for all probes."""
    eta = probe_noise(ids, t, tuple(z.shape[1:]), M, z.device, stream=SCORE_STREAM)
    out = []
    if mode == "tweedie":
        x0 = (z - sig * bundle.eps(z, t, y)) / A
        for m in range(M):
            xt = A * x0 + sig * eta[m]
            out.append(bundle.decode((xt - sig * bundle.eps(xt, t, y)) / A, chunk=4))
    else:
        delta = (radius * z.flatten(1).std(dim=1)).view(-1, 1, 1, 1)
        for m in range(M):
            x = z + delta * eta[m]
            out.append(bundle.decode((x - sig * bundle.eps(x, t, y)) / A, chunk=4))
    return out

@torch.no_grad()
def all_scores(bundle, phis, z, y, t, A, sig, ids, M, top_frac, mode="local", radius=None):
    """Calculate top-10% and all-patch scores for each feature model."""
    decoded = probe_images(bundle, z, y, t, A, sig, ids, M, mode,
                           radius if radius is not None else FINAL["probe_radius_rel"])
    out = {}
    for name, phi in phis.items():
        f = torch.stack([F.normalize(phi(d), dim=-1) for d in decoded])
        U = (f - f.mean(0)).pow(2).sum(-1).mean(0)  # One disagreement value per patch.
        Ul = to_latent(U, phi.grid, z.shape[-2:])
        mask, _ = build_mask_latent(Ul, top_frac)  # Select patches separately for each image.
        out[name] = (masked_values(U, mask, phi.grid, z.shape[-2:]).cpu(), U.mean(dim=1).cpu())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[128, 256, 512])
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--probe-mode", choices=["local", "tweedie"], default="local",
                    help="local uses nearby states; tweedie re-noises the predicted clean image")
    ap.add_argument("--probe-radius", type=float, default=None,
                    help="distance used by local probes")
    ap.add_argument("--band", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="fraction of sampling to score, for example --band 0.40 0.80")
    ap.add_argument("--points", choices=["dense", "six"], default="dense",
                    help="dense scores a range of steps; six scores only six fixed steps")
    a = ap.parse_args()

    p = PROTOCOL[a.resolution]
    radius = FINAL["probe_radius_rel"] if a.probe_radius is None else a.probe_radius
    if min(a.n, a.batch_size, a.shard_size) <= 0:
        ap.error("n, batch-size and shard-size must be positive")
    if not 0 <= radius < float("inf"):
        ap.error("probe-radius must be finite and nonnegative")
    if a.shard_size % a.batch_size or a.n % a.shard_size:
        raise SystemExit(f"shard {a.shard_size} must divide by batch {a.batch_size}, "
                         f"and n {a.n} by shard")
    out = Path(a.out or f"outputs/final_experiments/imagenet{a.resolution}/scores_dense")
    (out / "shards").mkdir(parents=True, exist_ok=True)
    manifest = out / "source_manifest.json"
    if not manifest.is_file() and any((out / "shards").glob("*.pt")):
        raise RuntimeError("existing scores have no source manifest; use a new output folder")
    sources = [__file__, PROJECT_ROOT / "runners/generate.py",
               *sorted((PROJECT_ROOT / "src/dino_uncertainty").rglob("*.py"))]
    write_or_verify_manifest(manifest, source_manifest(sources, root=PROJECT_ROOT))
    dev = torch.device(a.device)

    bundle, sd = load_backend(a.resolution, dev)
    sch = DSched(sd)
    steps = p["steps"]
    if a.band:
        lo, hi = float(a.band[0]), float(a.band[1])
        dense = [i for i in range(steps) if lo <= i / steps < hi]
        if not dense:
            raise SystemExit(f"band {lo}-{hi} selects no step at {steps} steps")
    else:
        lo, hi = DENSE_LO, DENSE_HI
        dense = dense_indices(steps)
    sparse = filter_indices(steps)  # Six fixed sampling points in addition to the dense band.
    want = ({i: i / steps for i in sorted(sparse.values())} if a.points == "six"
            else {i: i / steps for i in sorted(set(dense) | set(sparse.values()))})
    ac = sd.alphas_cumprod
    ts = [int(t) for t in sd.timesteps]

    from dino_uncertainty.features import build_extractor
    # Both feature models use the same probes so their scores can be compared fairly.
    phis = {"mae": build_extractor("mae", device=dev),
            "dino": build_extractor("dinov2crop", device=dev)}
    from dino_uncertainty.paths import starting_points
    SP = starting_points(a.resolution)
    X_T = torch.load(str(SP / "X_T.pth"), map_location="cpu", mmap=True)
    Y = torch.load(SP / "y.pth", map_location="cpu")
    if len(X_T) < a.n or len(Y) < a.n:
        raise SystemExit(f"need {a.n} starting points and labels, found {len(X_T)} and {len(Y)}")
    if tuple(X_T.shape[1:]) != p["latent"]:
        raise SystemExit(f"X_T is {tuple(X_T.shape[1:])}, protocol says {p['latent']}")

    write_run_settings(out / "provenance.json",
                {"resolution": a.resolution, "dataset": p["dataset"], "n": a.n,
                 "batch_size": a.batch_size, "shard_size": a.shard_size, "num_steps": steps,
                 "dense_band": [lo, hi], "dense_indices": dense,
                 "sparse_indices": {str(k): v for k, v in sparse.items()},
                 "scored_indices": sorted(want), "n_scored": len(want),
                 "scored_timesteps": {str(i): ts[i] for i in sorted(want)},
                 "encoders": {k: v.describe() for k, v in phis.items()},
                 "aggregations": {
                     "top10": "mean raw variance over the top-10% selected locations (spec, "
                              "matches the guidance loss)",
                     "all": "mean raw variance over every patch of that encoder"},
                 "score_keys": "<encoder>_<aggregation>, e.g. mae_top10, dino_all",
                 "M": FINAL["M"], "top_frac": FINAL["top_frac"],
                 "probe_mode": a.probe_mode, "probe_radius_rel": radius, "score_stream": SCORE_STREAM,
                 "guided": False, "fixed_start_X_T": str(SP / "X_T.pth"),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    print(f"[dense] {p['dataset']} n={a.n} bs={a.batch_size} | scoring {len(want)} steps "
          f"{sorted(want)[0]}..{sorted(want)[-1]} | encoders {list(phis)} x "
          f"aggregations [top10, all]", flush=True)

    t0, done, nsh = time.time(), 0, 0
    for lo in range(0, a.n, a.shard_size):
        hi = lo + a.shard_size
        sp = out / "shards" / f"{lo:06d}_{hi:06d}.pt"
        if sp.exists():
            validate_shard(sp, lo, hi, Y, want)
            print(f"[dense] reuse [{lo},{hi})", flush=True); continue
        if a.max_shards is not None and nsh >= a.max_shards:
            break
        acc = {f"{e}_{a}": {} for e in ("mae", "dino") for a in ("top10", "all")}
        for s in range(lo, hi, a.batch_size):
            e = s + a.batch_size
            ids = list(range(s, e))
            z = X_T[s:e].to(dev).float(); y = Y[s:e].to(dev).long()
            for i, t_ in enumerate(sch.timesteps):
                t = int(t_); al = ac[t]; A, sig = al.sqrt(), (1 - al).sqrt()
                if i in want:
                    sc = all_scores(bundle, phis, z, y, t, A, sig, ids,
                                    FINAL["M"], FINAL["top_frac"],
                                    mode=a.probe_mode, radius=radius)
                    for e, (tv, av) in sc.items():
                        acc[f"{e}_top10"].setdefault(i, []).append(tv)
                        acc[f"{e}_all"].setdefault(i, []).append(av)
                e_ = bundle.eps(z, t, y)
                z = sch.sched.step(e_, t, z).prev_sample
        atomic_save({"ids": list(range(lo, hi)), "labels": Y[lo:hi].clone(),
                     "fractions": {i: want[i] for i in sorted(want)},
                     **{k: {i: torch.cat(v) for i, v in d.items()} for k, d in acc.items()}},
                    sp)
        validate_shard(sp, lo, hi, Y, want)
        done += hi - lo; nsh += 1
        el = time.time() - t0
        atomic_json({"completed_to": hi, "n": a.n, "sec_per_image": el / max(done, 1),
                     "eta_h": (a.n - hi) * el / max(done, 1) / 3600,
                     "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30},
                    out / "progress.json")
        print(f"[dense] {hi}/{a.n} | {el/max(done,1):.2f}s/img | "
              f"ETA {(a.n-hi)*el/max(done,1)/3600:.2f}h | "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    expected = [out / "shards" / f"{lo:06d}_{lo + a.shard_size:06d}.pt"
                for lo in range(0, a.n, a.shard_size)]
    have = sum(path.is_file() for path in expected)
    atomic_json({"complete": have == len(expected), "shards": have,
                 "expected": len(expected)}, out / "manifest.json")
    print(f"[dense] {have}/{len(expected)} shards", flush=True)

if __name__ == "__main__":
    main()
