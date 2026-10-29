"""Generate guided samples using local feature disagreement."""
from __future__ import annotations
import argparse, math, os, time
from pathlib import Path

import torch

import dino_uncertainty.guidance.local_probe as EP
from dino_uncertainty.guidance.local_probe import guided_step, parse_indices
from dino_uncertainty.guidance.core import eps_cfg
from dino_uncertainty.config import PROTOCOL
from dino_uncertainty.paths import starting_points
from dino_uncertainty.provenance import source_manifest, write_or_verify_manifest, write_run_settings
from runners.generate import atomic_save, atomic_json, load_backend, DSched, to_u8

def validate_shard(path, lo, hi, res, labels, nwin, batch_size):
    rec = torch.load(path, map_location="cpu")
    cnt = hi - lo
    if torch.as_tensor(rec.get("ids")).tolist() != list(range(lo, hi)):
        raise RuntimeError(f"invalid image IDs in {path}")
    im, ft = rec.get("images"), rec.get("incep")
    if im is None or im.dtype != torch.uint8 or tuple(im.shape) != (cnt, 3, res, res):
        raise RuntimeError(f"invalid image tensor in {path}")
    if ft is None or tuple(ft.shape) != (cnt, 2048) or not torch.isfinite(ft).all():
        raise RuntimeError(f"invalid Inception features in {path}")
    if not torch.equal(torch.as_tensor(rec.get("labels")), labels[lo:hi]):
        raise RuntimeError(f"invalid class labels in {path}")
    st = rec.get("stats", [])
    if len(st) != cnt // batch_size * nwin:
        raise RuntimeError(f"invalid guidance record count in {path}: {len(st)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[128, 256, 512])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--decode-chunk", type=int, default=8)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    # Guidance settings.
    ap.add_argument("--probes", choices=["local", "tweedie"], default="tweedie")
    ap.add_argument("--space", default="mae",
                    help="feature model: mae or dino; other names are for comparison runs")
    ap.add_argument("--window", required=True,
                    help="guided steps, for example '40-45' or '40,43,45'")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--s0-rel", type=float, required=True)
    ap.add_argument("--s0-step-frac", type=float, required=True)
    ap.add_argument("--max-halvings", type=int, default=6)
    ap.add_argument("--all-tol", type=float, default=1e-3)
    ap.add_argument("--probe-radius", type=float, default=0.05)
    ap.add_argument("--probe-scale", type=float, default=1.0)
    ap.add_argument("--match-spread", action="store_true")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--probe-chunk", type=int, default=5)
    ap.add_argument("--heldout", action="store_true",
                    help="also measure disagreement with a second set of probes")
    ap.add_argument("--control", action="store_true",
                    help="use a random direction for a comparison run")
    ap.add_argument("--plain-gradient", action="store_true",
                    help="apply the first update without checking or reducing it")
    ap.add_argument("--probe-stream", type=int, default=None,
                    help="use a different random seed group for the probes")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if not math.isfinite(a.cfg):
        ap.error("cfg must be finite")
    if a.resolution == 128 and a.cfg != 1.0:
        ap.error("the ADM checkpoint has no unconditional class; cfg must be 1.0")

    res, p = a.resolution, PROTOCOL[a.resolution]
    steps, n = p["steps"], a.n
    if min(n, a.batch_size, a.shard_size, a.decode_chunk, a.probe_chunk, a.M) <= 0:
        ap.error("sample counts, batch sizes, chunk sizes and M must be positive")
    if not 0 < a.top_frac <= 1 or not 0 <= a.probe_radius < float("inf"):
        ap.error("top-frac must be in (0, 1] and probe-radius finite and nonnegative")
    if a.shard_size % a.batch_size:
        raise SystemExit(f"shard-size {a.shard_size} must be a multiple of batch-size {a.batch_size}")
    if n % a.shard_size:
        raise SystemExit(f"n {n} must be a multiple of shard-size {a.shard_size}")
    lo0, hi0 = a.lo, (a.hi if a.hi is not None else n)
    if lo0 % a.shard_size or hi0 % a.shard_size or not 0 <= lo0 < hi0 <= n:
        raise SystemExit(f"--lo/--hi must be shard-aligned inside 0..{n}")

    win = sorted(parse_indices(a.window, steps))
    tag = Path(a.out).name if a.out else f"guided_exact_{a.window}"
    out = Path(a.out or f"outputs/final_experiments/imagenet{res}/{tag}")
    (out / "shards").mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    EP.CHUNK["probes"] = a.probe_chunk

    replay = None
    if a.replay:
        import json
        src = json.loads(Path(a.replay).read_text())
        replay = {f"{i}_{r['step']}": s for r in src["records"]
                  for i, s, ok in zip(r["ids"], r["s_accepted"], r["accepted"]) if ok}

    bundle, sd = load_backend(res, dev)
    sch = DSched(sd)
    ac = sd.alphas_cumprod
    ts = [int(t) for t in sch.timesteps]
    from dino_uncertainty.features import build_extractor, SPEC_ALIASES
    # Check feature-model names before starting a long run.
    LEGACY = {"mae": "mae", "dino": "dinov2crop"}
    spec = LEGACY.get(a.space, a.space)
    if spec not in SPEC_ALIASES and spec not in LEGACY.values():
        raise SystemExit(f"unknown --space {a.space!r}; choose mae, dino, or one of "
                         f"{sorted(SPEC_ALIASES)}")
    ex = build_extractor(spec, device=dev)
    from dino_uncertainty.metrics.fid import inception_features, load_inception
    inception = load_inception(dev)

    SP = starting_points(res)
    X_T = torch.load(str(SP / "X_T.pth"), map_location="cpu", mmap=True)
    Y = torch.load(SP / "y.pth", map_location="cpu")
    if len(X_T) < n or len(Y) < n:
        raise SystemExit(f"need {n} starting points and labels, found {len(X_T)} and {len(Y)}")
    if tuple(X_T.shape[1:]) != p["latent"]:
        raise SystemExit(f"X_T is {tuple(X_T.shape[1:])}, protocol says {p['latent']}")

    write_or_verify_manifest(out / "source_manifest.json", source_manifest([
        __file__, "src/dino_uncertainty/guidance/local_probe.py"]))
    prov = {
        "resolution": res, "dataset": p["dataset"], "n": n, "num_steps": steps,
        "batch_size": a.batch_size, "shard_size": a.shard_size, "decode_chunk": a.decode_chunk,
        "guidance_indices": win, "guidance_timesteps": [ts[i] for i in win],
        "guidance_alpha_bar": [float(ac[ts[i]]) for i in win],
        "probes": a.probes, "space": a.space, "top_frac": a.top_frac, "M": a.M,
        "s0_rel": a.s0_rel, "s0_step_frac": a.s0_step_frac, "max_halvings": a.max_halvings,
        "all_tol": a.all_tol, "probe_radius": a.probe_radius, "probe_scale": a.probe_scale,
        "match_spread": a.match_spread, "cfg": a.cfg, "probe_chunk": a.probe_chunk,
        "heldout": a.heldout, "control": a.control, "replay": a.replay,
        "accept_test": not a.plain_gradient, "probe_stream": a.probe_stream,
        "encoder": ex.describe(), "fixed_start_X_T": str(SP / "X_T.pth"),
        "unguided_baseline": f"{out.parent / 'pool'} (not regenerated)",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_run_settings(out / "provenance.json", prov)
    print(f"[exact10k] {p['dataset']} {tag}: n={n} steps={steps} batch={a.batch_size}", flush=True)
    print(f"[exact10k] window {win} -> timesteps {prov['guidance_timesteps']}", flush=True)
    print(f"[exact10k] probes={a.probes} match_spread={a.match_spread} top_frac={a.top_frac} "
          f"s0_rel={a.s0_rel} s0_step_frac={a.s0_step_frac} heldout={a.heldout}", flush=True)

    t0, done, nsh = time.time(), 0, 0
    if (lo0, hi0) != (0, n):
        print(f"[exact10k] this job covers IDs {lo0}..{hi0 - 1} of {n}", flush=True)
    for lo in range(lo0, hi0, a.shard_size):
        hi = lo + a.shard_size
        sp = out / "shards" / f"{lo:06d}_{hi:06d}.pt"
        if sp.exists():
            validate_shard(sp, lo, hi, res, Y, len(win), a.batch_size)
            print(f"[exact10k] reuse shard [{lo},{hi})", flush=True); continue
        if a.max_shards is not None and nsh >= a.max_shards:
            break
        imgs, feats, allst = [], [], []
        for s in range(lo, hi, a.batch_size):
            e = s + a.batch_size
            ids = list(range(s, e))
            z = X_T[s:e].to(dev).float(); y = Y[s:e].to(dev).long()
            for i, t_ in enumerate(sch.timesteps):
                t = int(t_)
                with torch.no_grad():
                    e0 = eps_cfg(bundle, z, t, y, a.cfg)
                if i in win:
                    z, e0, rec = guided_step(bundle, ex, sch, z, y, ids, t, i,
                                             mode=a.probes, top_frac=a.top_frac, M=a.M,
                                             s0_rel=a.s0_rel, s0_step_frac=a.s0_step_frac,
                                             max_halvings=a.max_halvings, all_tol=a.all_tol,
                                             e0=e0, control=a.control, replay=replay,
                                             accept_test=not a.plain_gradient,
                                             probe_stream=a.probe_stream,
                                             cfg=a.cfg, probe_radius=a.probe_radius,
                                             probe_scale=a.probe_scale,
                                             match_spread=a.match_spread, heldout=a.heldout)
                    allst.append(rec)
                z = sch.sched.step(e0, t, z).prev_sample
            with torch.no_grad():
                u8 = to_u8(bundle.decode(z, chunk=a.decode_chunk))
            feats.append(inception_features(inception, u8, dev))
            imgs.append(u8.cpu())
        atomic_save({"ids": list(range(lo, hi)), "images": torch.cat(imgs),
                     "incep": torch.cat(feats), "labels": Y[lo:hi].clone(),
                     "stats": allst}, sp)
        validate_shard(sp, lo, hi, res, Y, len(win), a.batch_size)
        done += hi - lo; nsh += 1
        el = time.time() - t0
        eta = (hi0 - hi) * el / max(done, 1) / 3600
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        rel = [v for it in allst for v in it["rel_edit"]]
        acc = [v for it in allst for v in it["accepted"]]
        atomic_json({"completed_to": hi, "range": [lo0, hi0], "n": n,
                     "sec_per_image": el / max(done, 1), "elapsed_h": el / 3600, "eta_h": eta,
                     "peak_gpu_gib": mem,
                     "rel_edit_mean": sum(rel) / max(len(rel), 1),
                     "accept_rate": sum(acc) / max(len(acc), 1)},
                    out / f"progress_{lo0:06d}_{hi0:06d}.json")
        print(f"[exact10k] {hi}/{hi0} (arm n={n}) | {el/max(done,1):.2f}s/img | ETA {eta:.2f}h | "
              f"peak {mem:.1f} GiB | rel_edit {sum(rel)/max(len(rel),1):.4f} | "
              f"acc {sum(acc)/max(len(acc),1):.3f}", flush=True)

    expected = [out / "shards" / f"{q:06d}_{q + a.shard_size:06d}.pt"
                for q in range(0, n, a.shard_size)]
    present = sum(q.is_file() for q in expected)
    atomic_json({**prov, "shards_present": present, "shards_expected": len(expected),
                 "complete": present == len(expected)}, out / "manifest.json")
    print(f"[exact10k] {present}/{len(expected)} shards present", flush=True)

if __name__ == "__main__":
    main()
