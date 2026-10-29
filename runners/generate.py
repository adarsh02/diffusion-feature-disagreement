"""Generate unguided ImageNet samples and optional filtering scores."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import torch
import torch.nn.functional as F

from dino_uncertainty.config import FINAL, PROTOCOL, window_indices, filter_indices
from dino_uncertainty.guidance.core import masked_values, probe_noise, to_latent, build_mask_latent
from dino_uncertainty.paths import starting_points
from dino_uncertainty.provenance import source_manifest, write_or_verify_manifest, write_run_settings

SCORE_STREAM = 7717  # A separate random seed group for filtering scores.

def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp); os.replace(tmp, path)

def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n"); os.replace(tmp, path)

def load_backend(res: int, device, steps: int | None = None):
    """Load the model and DDIM sampling steps for one image size."""
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    p = dict(PROTOCOL[res])
    if steps is not None:
        p["steps"] = int(steps)
    if p["backend"] == "adm":
        from dino_uncertainty.backends.adm import load_adm_bundle, load_adm_scheduler
        bundle = load_adm_bundle(device)
        src = load_adm_scheduler()
    else:
        from dino_uncertainty.backends.uvit import load_uvit_bundle, load_uvit_scheduler
        bundle = load_uvit_bundle(device, image_size=res)
        src = load_uvit_scheduler()
    sd = DDIMScheduler.from_config(src.config)
    sd.set_timesteps(p["steps"])
    if len(sd.timesteps) != p["steps"]:
        raise RuntimeError(f"asked {p['steps']} steps, scheduler built {len(sd.timesteps)}")
    return bundle, sd

class DSched:
    """Give the sampling loop a small, common scheduler interface."""
    def __init__(self, sched):
        self.sched = sched; self.timesteps = sched.timesteps

def to_u8(x):
    return ((x.clamp(-1, 1) / 2 + .5) * 255).round().to(torch.uint8)

def validate_shard(path, lo, hi, resolution, arm, labels, points, window, batch_size):
    """Check a saved output file before reusing it."""
    record = torch.load(path, map_location="cpu")
    count = hi - lo
    if torch.as_tensor(record.get("ids")).tolist() != list(range(lo, hi)):
        raise RuntimeError(f"invalid image IDs in {path}")
    images = record.get("images")
    features = record.get("incep")
    saved_labels = torch.as_tensor(record.get("labels"))
    if images is None or images.dtype != torch.uint8 or tuple(images.shape) != (
        count,
        3,
        resolution,
        resolution,
    ):
        raise RuntimeError(f"invalid image tensor in {path}")
    if features is None or tuple(features.shape) != (count, 2048) or not torch.isfinite(features).all():
        raise RuntimeError(f"invalid Inception features in {path}")
    if not torch.equal(saved_labels, labels[lo:hi]):
        raise RuntimeError(f"invalid class labels in {path}")
    if arm == "unguided":
        # Unguided samples do not contain filtering scores or guidance details.
        return
    if arm == "pool":
        scores = record.get("scores", {})
        if set(scores) != set(points):
            raise RuntimeError(f"invalid score points in {path}")
        if any(torch.as_tensor(value).shape != (count,) or not torch.isfinite(value).all()
               for value in scores.values()):
            raise RuntimeError(f"invalid uncertainty scores in {path}")
    else:
        stats = record.get("stats", [])
        expected_records = count // batch_size * len(window)
        if len(stats) != expected_records:
            raise RuntimeError(
                f"invalid guidance record count in {path}: {len(stats)} != {expected_records}"
            )
        for index in window:
            observed = [image_id for item in stats if item.get("idx") == index
                        for image_id in item.get("image_ids", [])]
            if observed != list(range(lo, hi)):
                raise RuntimeError(f"missing or duplicated guided IDs at index {index} in {path}")
        if any(item.get("cells", 0) <= 0 or
               len(item.get("achieved_edit", [])) != batch_size or
               len(item.get("skipped", [])) != batch_size
               for item in stats):
            raise RuntimeError(f"invalid guidance statistics in {path}")

@torch.no_grad()
def score_point(bundle, phi, z, y, t, A, sig, ids, M, top_frac,
                probe_radius=FINAL["probe_radius_rel"]):
    """Calculate one filtering score for each image at the current step."""
    delta = (probe_radius * z.flatten(1).std(dim=1)).view(-1, 1, 1, 1)
    eta = probe_noise(ids, t, tuple(z.shape[1:]), M, z.device, stream=SCORE_STREAM)
    f = []
    for m in range(M):
        x = z + delta * eta[m]
        b = (x - sig * bundle.eps(x, t, y)) / A
        f.append(F.normalize(phi(bundle.decode(b, chunk=4)), dim=-1))
    f = torch.stack(f)
    U = (f - f.mean(0)).pow(2).sum(-1).mean(0)  # One disagreement value per patch.
    Ul = to_latent(U, phi.grid, z.shape[-2:])
    mask, _ = build_mask_latent(Ul, top_frac)
    return masked_values(U, mask, phi.grid, z.shape[-2:]).cpu()

def run_pool(bundle, phi, sch, X, Y, ids, M, top_frac, pts, dev,
             probe_radius=FINAL["probe_radius_rel"]):
    """Generate unguided samples and score them at six sampling steps."""
    z = X.clone()
    ac = sch.sched.alphas_cumprod.to(z.device)
    want = {i: p for p, i in pts.items()}
    scores = {}
    for i, t_ in enumerate(sch.timesteps):
        t = int(t_)
        a = ac[t]; A, sig = a.sqrt(), (1 - a).sqrt()
        if i in want:
            scores[want[i]] = score_point(bundle, phi, z, Y, t, A, sig, ids, M, top_frac,
                                          probe_radius)
        with torch.no_grad():
            e = bundle.eps(z, t, Y)
        z = sch.sched.step(e, t, z).prev_sample
    return z, scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[128, 256, 512])
    ap.add_argument("--arm", required=True, choices=["pool", "unguided"],
                    help="pool saves filtering scores; unguided only generates images")
    ap.add_argument("--window", default=None,
                    help="optional inclusive step range such as '22-28'")
    ap.add_argument("--probe-radius", type=float, default=None,
                    help="probe distance as a fraction of the current state's standard deviation")
    ap.add_argument("--M", type=int, default=None, help="number of probes")
    ap.add_argument("--sigma-blur", type=float, default=None,
                    help="blur size on the model's internal image grid; 0 disables it")
    ap.add_argument("--top-frac", type=float, default=None,
                    help="fraction of the highest-scoring locations to use")
    ap.add_argument("--space", choices=["mae", "dino"], default="mae",
                    help="feature model used to measure disagreement")
    ap.add_argument("--kappa", type=float, default=None,
                    help="guidance strength used by older comparison runs")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="stop after this many output files")
    ap.add_argument("--lo", type=int, default=0,
                    help="first image ID; use with --hi to split work across GPUs")
    ap.add_argument("--hi", type=int, default=None, help="one past the last image ID")
    ap.add_argument("--steps", type=int, default=None,
                    help="override the number of sampling steps")
    a = ap.parse_args()
    if any(v is not None for v in (a.window, a.sigma_blur, a.kappa)):
        ap.error("--window, --sigma-blur and --kappa do not apply to unguided generation; "
                 "use runners.generate_local_probe for guidance")

    res, arm = a.resolution, a.arm
    p = dict(PROTOCOL[res])
    if a.steps is not None:
        # Keep the later step calculations in sync with the model scheduler.
        p["steps"] = int(a.steps)
    n = a.n if a.n is not None else (12000 if arm == "pool" else 10000)
    kappa = FINAL["kappa"] if a.kappa is None else float(a.kappa)
    top_frac = FINAL["top_frac"] if a.top_frac is None else float(a.top_frac)
    sigma_blur = FINAL["sigma_blur"] if a.sigma_blur is None else float(a.sigma_blur)
    probe_r = FINAL["probe_radius_rel"] if a.probe_radius is None else float(a.probe_radius)
    M = FINAL["M"] if a.M is None else int(a.M)
    if min(n, a.batch_size, a.shard_size) <= 0:
        ap.error("n, batch-size and shard-size must be positive")
    if M < 1 or not 0 < top_frac <= 1 or not 0 <= probe_r < float("inf"):
        ap.error("M must be positive, top-frac in (0, 1], and probe-radius finite and nonnegative")
    if a.shard_size % a.batch_size:
        raise SystemExit(f"shard-size {a.shard_size} must be a multiple of batch-size "
                         f"{a.batch_size} so every batch has the same size.")
    if n % a.shard_size:
        raise SystemExit(f"n {n} must be a multiple of shard-size {a.shard_size}")
    lo0, hi0 = int(a.lo), int(a.hi if a.hi is not None else n)
    if lo0 % a.shard_size or hi0 % a.shard_size or not 0 <= lo0 < hi0 <= n:
        raise SystemExit(f"--lo/--hi must be shard-aligned and inside [0,{n}]; "
                         f"got {lo0}..{hi0} with shard-size {a.shard_size}")

    out = Path(a.out or f"outputs/final_experiments/imagenet{res}/{arm}")
    (out / "shards").mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    bundle, sd = load_backend(res, dev, a.steps)
    sch = DSched(sd)
    # Use the requested number of steps when one was supplied.
    steps = a.steps or p["steps"]
    if a.window:
        lo_w, hi_w = (int(x) for x in a.window.split("-"))
        win = set(range(lo_w, hi_w + 1))  # Include both ends of the range.
        if not win or max(win) >= steps:
            raise SystemExit(f"window {a.window} outside 0..{steps - 1}")
    else:
        win = set(window_indices(steps))
    pts = filter_indices(steps)
    ac = sd.alphas_cumprod
    ts = [int(t) for t in sd.timesteps]

    from dino_uncertainty.features import build_extractor
    phi = (build_extractor("mae" if a.space == "mae" else "dinov2crop", device=dev)
           if arm == "pool" else None)

    sources = [
        __file__,
        "src/dino_uncertainty/config.py",
        "src/dino_uncertainty/guidance/core.py",
        f"src/dino_uncertainty/features/{'mae' if a.space == 'mae' else 'dinov2'}.py",
        "src/dino_uncertainty/backends/adm.py" if res == 128 else "src/dino_uncertainty/backends/uvit.py",
    ]
    write_or_verify_manifest(out / "source_manifest.json", source_manifest(sources))

    SP = starting_points(res)
    X_T = torch.load(str(SP / "X_T.pth"), map_location="cpu", mmap=True)
    Y = torch.load(SP / "y.pth", map_location="cpu")
    if len(X_T) < n or len(Y) < n:
        raise SystemExit(f"need {n} starting points and labels, found {len(X_T)} and {len(Y)}")
    if tuple(X_T.shape[1:]) != p["latent"]:
        raise SystemExit(f"X_T is {tuple(X_T.shape[1:])}, protocol says {p['latent']}")

    from dino_uncertainty.metrics.fid import inception_features, load_inception
    inception = load_inception(dev)

    prov = {
        "arm": arm, "resolution": res, "dataset": p["dataset"], "backend": p["backend"],
        "n": n, "batch_size": a.batch_size, "shard_size": a.shard_size,
        "num_steps": steps, "executed_timesteps": ts,
        "image_ids": [0, n - 1], "id_rule": "contiguous 0..n-1 of De Vita's fixed X_T/y",
        "fixed_start_X_T": str(SP / "X_T.pth"), "fixed_start_y": str(SP / "y.pth"),
        "final_settings": {**dict(FINAL), "kappa": kappa, "top_frac": top_frac,
                           "sigma_blur": sigma_blur, "probe_radius_rel": probe_r, "M": M,
                           "space": a.space},
        "window_overridden": a.window is not None,
        "kappa_overridden": a.kappa is not None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "guidance_indices": sorted(win) if arm == "guided" else [],
        "guidance_timesteps": [ts[i] for i in sorted(win)] if arm == "guided" else [],
        "guidance_alpha_bar": [float(ac[ts[i]]) for i in sorted(win)] if arm == "guided" else [],
        "score_points": {str(k): v for k, v in pts.items()} if arm == "pool" else {},
        "score_timesteps": {str(k): ts[v] for k, v in pts.items()} if arm == "pool" else {},
        "score_alpha_bar": ({str(k): float(ac[ts[v]]) for k, v in pts.items()}
                            if arm == "pool" else {}),
        "score_stream": SCORE_STREAM,
        "encoder": phi.describe() if phi is not None else None,
        "source_manifest": str(out / "source_manifest.json"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_run_settings(out / "provenance.json", prov)
    print(f"[final_gen] {p['dataset']} {arm}: n={n} steps={steps} batch={a.batch_size}", flush=True)
    if arm == "guided":
        print(f"[final_gen] guidance indices {sorted(win)} -> timesteps "
              f"{prov['guidance_timesteps']}", flush=True)
    else:
        print(f"[final_gen] score points {prov['score_points']} -> timesteps "
              f"{prov['score_timesteps']}", flush=True)

    t0, done, nsh = time.time(), 0, 0
    if (lo0, hi0) != (0, n):
        print(f"[final_gen] this job covers IDs {lo0}..{hi0 - 1} of {n}", flush=True)
    for lo in range(lo0, hi0, a.shard_size):
        hi = lo + a.shard_size
        sp = out / "shards" / f"{lo:06d}_{hi:06d}.pt"
        if sp.exists():
            validate_shard(sp, lo, hi, res, arm, Y, pts, win, a.batch_size)
            print(f"[final_gen] reuse shard [{lo},{hi})", flush=True); continue
        if a.max_shards is not None and nsh >= a.max_shards:
            print(f"[final_gen] stopping after {nsh} shards (--max-shards)", flush=True); break
        imgs, feats, scores, allst = [], [], {}, []
        for s in range(lo, hi, a.batch_size):
            e = s + a.batch_size
            ids = list(range(s, e))
            x = X_T[s:e].to(dev).float(); y = Y[s:e].to(dev).long()
            if arm == "unguided":
                z = x
                with torch.no_grad():
                    for i, t_ in enumerate(sch.timesteps):
                        t = int(t_)
                        z = sch.sched.step(bundle.eps(z, t, y), t, z).prev_sample
            else:
                z, sc = run_pool(bundle, phi, sch, x, y, ids, M, top_frac, pts, dev,
                                 probe_radius=probe_r)
                for k, v in sc.items():
                    scores.setdefault(k, []).append(v)
            with torch.no_grad():
                u8 = to_u8(bundle.decode(z, chunk=4))
            feats.append(inception_features(inception, u8, dev))
            imgs.append(u8.cpu())
        rec = {"ids": list(range(lo, hi)), "images": torch.cat(imgs),
               "incep": torch.cat(feats), "labels": Y[lo:hi].clone()}
        if arm == "unguided":
            pass
        elif arm == "pool":
            rec["scores"] = {k: torch.cat(v) for k, v in scores.items()}
        else:
            rec["stats"] = allst
        atomic_save(rec, sp)
        validate_shard(sp, lo, hi, res, arm, Y, pts, win, a.batch_size)
        done += hi - lo; nsh += 1
        el = time.time() - t0
        # Estimate time to the end of this job's assigned image range.
        eta = (hi0 - hi) * el / max(done, 1) / 3600
        mem = torch.cuda.max_memory_allocated() / 2**30
        atomic_json({"completed_to": hi, "range": [lo0, hi0], "n": n,
                     "sec_per_image": el / max(done, 1), "elapsed_h": el / 3600,
                     "eta_h": eta, "peak_gpu_gib": mem},
                    out / f"progress_{lo0:06d}_{hi0:06d}.json")
        print(f"[final_gen] {hi}/{hi0} (arm n={n}) | {el/max(done,1):.2f}s/img | "
              f"ETA {eta:.2f}h | peak {mem:.1f} GiB", flush=True)
    expected = [out / "shards" / f"{lo:06d}_{lo + a.shard_size:06d}.pt"
                for lo in range(0, n, a.shard_size)]
    present = sum(path.is_file() for path in expected)
    atomic_json({**prov, "shards_present": present, "shards_expected": len(expected),
                 "complete": all(path.is_file() for path in expected)},
                out / "manifest.json")
    print(f"[final_gen] {present}/{len(expected)} shards present", flush=True)

if __name__ == "__main__":
    main()
