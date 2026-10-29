"""Run a small guidance experiment for checking different settings."""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path

import torch

import dino_uncertainty.guidance.local_probe as EP
from dino_uncertainty.guidance.local_probe import guided_step, parse_indices
from dino_uncertainty.guidance.core import eps_cfg
from dino_uncertainty.config import PROTOCOL
from dino_uncertainty.paths import starting_points
from dino_uncertainty.provenance import source_manifest
from runners.generate import atomic_json, load_backend, DSched, to_u8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=256, choices=[128, 256, 512])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--lo", type=int, default=12000,
                    help="first image ID; the default does not overlap the main sample pool")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--probes", choices=["local", "tweedie"], default="local")
    ap.add_argument("--space", choices=["mae", "dino"], default="mae",
                    help="feature model used to measure disagreement")
    ap.add_argument("--top-frac", type=float, default=0.10,
                    help="fraction of the highest-scoring patches to use; 1.0 uses all patches")
    ap.add_argument("--window", default="25-30",
                    help="first and last guided steps, both included")
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--s0-rel", type=float, default=0.005)
    ap.add_argument("--s0-step-frac", type=float, default=0.10)
    ap.add_argument("--max-halvings", type=int, default=6)
    ap.add_argument("--all-tol", type=float, default=1e-3,
                    help="allowed increase across all patches before an update is skipped")
    ap.add_argument("--max-doublings", type=int, default=0,
                    help="number of times a successful update may be doubled; 0 disables this")
    ap.add_argument("--probe-radius", type=float, default=0.05,
                    help="distance used by local probes")
    ap.add_argument("--probe-scale", type=float, default=1.0,
                    help="distance multiplier used by Tweedie probes")
    ap.add_argument("--match-spread", action="store_true",
                    help="give local and Tweedie probes the same spread")
    ap.add_argument("--cfg", type=float, default=1.0,
                    help="classifier-free guidance weight; 1.0 = single conditional pass")
    ap.add_argument("--control", action="store_true",
                    help="use a random direction instead of the score gradient")
    ap.add_argument("--replay", default=None,
                    help="saved update sizes from a matched comparison run")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="outputs/exact_probe")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--baseline-cache", default=None,
                    help="saved unguided samples for these image IDs; created if missing")
    ap.add_argument("--probe-chunk", type=int, default=1,
                    help="probes handled together; larger values are faster but use more memory")
    ap.add_argument("--no-heldout", dest="heldout", action="store_false",
                    help="skip the second probe set that is used only for reporting")
    a = ap.parse_args()

    if not math.isfinite(a.cfg):
        ap.error("cfg must be finite")
    if a.resolution == 128 and a.cfg != 1.0:
        ap.error("the ADM checkpoint has no unconditional class; cfg must be 1.0")

    res, p = a.resolution, PROTOCOL[a.resolution]
    steps = p["steps"]
    win = set(parse_indices(a.window, steps))
    out = Path(a.out) / a.tag
    (out).mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    replay = None
    if a.replay:
        src = json.loads(Path(a.replay).read_text())
        replay = {f"{i}_{r['step']}": s
                  for r in src["records"]
                  for i, s, ok in zip(r["ids"], r["s_accepted"], r["accepted"]) if ok}
        print(f"[exact] replaying {len(replay)} accepted updates from {a.replay}", flush=True)

    EP.CHUNK["probes"] = a.probe_chunk
    print(f"[exact] probe_chunk={a.probe_chunk}  heldout={a.heldout}", flush=True)
    bundle, sd = load_backend(res, dev)
    sch = DSched(sd)
    from dino_uncertainty.features import build_extractor
    ex = build_extractor("mae" if a.space == "mae" else "dinov2crop", device=dev)
    if a.space == "dino" and not ex.describe().get("num_register_tokens"):
        raise SystemExit(f"DINOv2 must be the register variant, got {ex.describe().get('model_id')}")

    SP = starting_points(res)
    X_T = torch.load(str(SP / "X_T.pth"), map_location="cpu", mmap=True)
    Y = torch.load(SP / "y.pth", map_location="cpu")
    if a.lo + a.n > len(X_T):
        raise SystemExit(f"need IDs {a.lo}..{a.lo+a.n-1}, only {len(X_T)} starting points")

    atomic_json({"resolution": res, "dataset": p["dataset"], "n": a.n, "lo": a.lo,
                 "ids": [a.lo + i for i in range(a.n)], "num_steps": steps,
                 "window": sorted(win), "probes": a.probes, "space": a.space,
                 "top_frac": a.top_frac, "M": a.M,
                 "s0_rel": a.s0_rel, "s0_step_frac": a.s0_step_frac,
                 "max_halvings": a.max_halvings, "max_doublings": a.max_doublings,
                 "all_tol": a.all_tol, "probe_radius": a.probe_radius, "probe_scale": a.probe_scale,
                 "match_spread": a.match_spread, "cfg": a.cfg,
                 "probe_chunk": a.probe_chunk, "heldout": a.heldout,
                 "control": a.control, "replay": a.replay, "encoder": ex.describe(),
                 "batch_size": a.batch_size,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "sources": source_manifest([__file__, "src/dino_uncertainty/guidance/local_probe.py"])},
                out / "provenance.json")

    print(f"[exact] {p['dataset']} ids {a.lo}..{a.lo+a.n-1} | space={a.space} probes={a.probes} "
          f"top_frac={a.top_frac} window={sorted(win)} M={a.M} s0_rel={a.s0_rel} "
          f"radius={a.probe_radius} scale={a.probe_scale} match_spread={a.match_spread} "
          f"control={a.control} replay={bool(a.replay)}", flush=True)

    baseline_settings = {"resolution": res, "cfg": a.cfg, "num_steps": steps,
                         "starting_points": str(SP.resolve()),
                         "sources": source_manifest([
                             __file__, "runners/generate.py",
                             f"src/dino_uncertainty/backends/{'adm' if res == 128 else 'uvit'}.py",
                             "src/dino_uncertainty/guidance/core.py"])}
    cache = Path(a.baseline_cache) if a.baseline_cache else None
    cached = None
    if cache is not None and cache.exists():
        rec = torch.load(cache, map_location="cpu")
        if rec.get("settings") != baseline_settings:
            raise SystemExit(f"{cache} has unverified or different sampling settings; "
                             "use a new baseline cache file")
        if [int(v) for v in rec["ids"]] != [a.lo + i for i in range(a.n)]:
            raise SystemExit(f"{cache} holds different image IDs; use a separate cache file")
        cached = rec["images"]
        print(f"[exact] reusing unguided baseline {tuple(cached.shape)} from {cache}", flush=True)

    recs, G, Uimgs, t0 = [], [], [], time.time()
    for s in range(0, a.n, a.batch_size):
        e = min(s + a.batch_size, a.n)
        ids = [a.lo + i for i in range(s, e)]
        z = X_T[a.lo + s:a.lo + e].to(dev).float()
        y = Y[a.lo + s:a.lo + e].to(dev).long()
        zu = None if cached is not None else z.clone()
        for i, t_ in enumerate(sch.timesteps):
            t = int(t_)
            with torch.no_grad():
                e0 = eps_cfg(bundle, z, t, y, a.cfg)
                eu = None if zu is None else eps_cfg(bundle, zu, t, y, a.cfg)
            if i in win:
                z, e0, rec = guided_step(bundle, ex, sch, z, y, ids, t, i,
                                         mode=a.probes, top_frac=a.top_frac, M=a.M,
                                         s0_rel=a.s0_rel, s0_step_frac=a.s0_step_frac,
                                         max_halvings=a.max_halvings, all_tol=a.all_tol,
                                         e0=e0, control=a.control, replay=replay, cfg=a.cfg,
                                         probe_radius=a.probe_radius,
                                         probe_scale=a.probe_scale,
                                         match_spread=a.match_spread,
                                         max_doublings=a.max_doublings, heldout=a.heldout)
                recs.append(rec)
            z = sch.sched.step(e0, t, z).prev_sample
            if zu is not None:
                zu = sch.sched.step(eu, t, zu).prev_sample
        with torch.no_grad():
            G.append(to_u8(bundle.decode(z, chunk=a.chunk)).cpu())
            if zu is not None:
                Uimgs.append(to_u8(bundle.decode(zu, chunk=a.chunk)).cpu())
        el = time.time() - t0
        print(f"[exact] {e}/{a.n} | {el/e:.2f}s/img | "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)

    G = torch.cat(G)
    Uimgs = cached if cached is not None else torch.cat(Uimgs)
    if cache is not None and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": [a.lo + i for i in range(a.n)], "images": Uimgs,
                    "settings": baseline_settings}, cache)
        print(f"[exact] wrote unguided baseline cache {cache}", flush=True)
    torch.save({"ids": [a.lo + i for i in range(a.n)], "images": G}, out / "guided.pt")
    torch.save({"ids": [a.lo + i for i in range(a.n)], "images": Uimgs}, out / "unguided.pt")

    nprop = sum(len(r["ids"]) for r in recs)
    nacc = sum(sum(r["accepted"]) for r in recs)
    dmask = [b - af for r in recs for b, af, ok in
             zip(r["masked_var_before"], r["masked_var_after"], r["accepted"]) if ok]
    dheld = [b - af for r in recs for b, af, ok in
             zip(r["heldout_masked_before"], r["heldout_masked_after"], r["accepted"]) if ok]
    rmask = [(b - af) / max(b, 1e-12) for r in recs for b, af, ok in
             zip(r["masked_var_before"], r["masked_var_after"], r["accepted"]) if ok]
    rheld = [(b - af) / max(b, 1e-12) for r in recs for b, af, ok in
             zip(r["heldout_masked_before"], r["heldout_masked_after"], r["accepted"]) if ok]
    diff = (G.float() - Uimgs.float()).abs()
    summary = {
        "tag": a.tag, "probes": a.probes, "space": a.space, "top_frac": a.top_frac,
        "window": sorted(win), "s0_rel": a.s0_rel, "s0_step_frac": a.s0_step_frac,
        "probe_radius": a.probe_radius, "probe_scale": a.probe_scale,
        "match_spread": a.match_spread, "cfg": a.cfg, "M": a.M,
        "max_halvings": a.max_halvings, "max_doublings": a.max_doublings,
        "all_tol": a.all_tol,
        "mean_doublings": float(sum(v for r in recs for v in r.get("doublings", [0]))
                                / max(nprop, 1)),
        "mean_probe_spread": float(sum(r["probe_spread"] for r in recs) / max(len(recs), 1)),
        "mean_sigma_t": float(sum(r["sigma_t"] for r in recs) / max(len(recs), 1)),
        "mean_U0_masked": float(sum(r["U0_masked_mean"] for r in recs) / max(len(recs), 1)),
        "control": a.control, "replay": a.replay,
        "proposals": nprop, "accepted": nacc, "skipped": nprop - nacc,
        "accept_rate": nacc / max(nprop, 1),
        "mean_trials": sum(sum(r["trials"]) for r in recs) / max(nprop, 1),
        "mean_d_masked_var_accepted": float(sum(dmask) / max(len(dmask), 1)),
        "mean_rel_d_masked_var_accepted": float(sum(rmask) / max(len(rmask), 1)),
        "mean_d_heldout_masked_accepted": float(sum(dheld) / max(len(dheld), 1)),
        "mean_rel_d_heldout_masked_accepted": float(sum(rheld) / max(len(rheld), 1)),
        "mean_rel_edit": float(sum(v for r in recs for v in r["rel_edit"]) / max(nprop, 1)),
        "mean_abs_rgb_255": float(diff.mean()),
        "max_abs_rgb_255": float(diff.max()),
        "frac_pixels_moved": float((diff > 0.5).float().mean()),
        "per_image_abs_rgb": [float(v) for v in diff.flatten(1).mean(1)],
        "runtime_sec": time.time() - t0, "sec_per_image": (time.time() - t0) / a.n,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    }
    atomic_json(summary, out / "summary.json")
    atomic_json({"records": recs}, out / "records.json")
    print(f"\n[exact] {a.tag}: {nacc}/{nprop} accepted ({100*nacc/max(nprop,1):.1f}%), "
          f"mean {summary['mean_trials']:.2f} trials", flush=True)
    print(f"[exact]   same-noise masked var  {summary['mean_rel_d_masked_var_accepted']*100:+.3f}% "
          f"| held-out {summary['mean_rel_d_heldout_masked_accepted']*100:+.3f}%", flush=True)
    print(f"[exact]   probe spread {summary['mean_probe_spread']:.4f} "
          f"(sigma_t {summary['mean_sigma_t']:.3f}), U0_masked {summary['mean_U0_masked']:.3e}",
          flush=True)
    print(f"[exact]   |dRGB| {summary['mean_abs_rgb_255']:.3f}/255, "
          f"{summary['frac_pixels_moved']*100:.1f}% of pixels moved, "
          f"peak {summary['peak_gpu_gib']:.1f} GiB", flush=True)
    print(f"[exact] wrote {out}", flush=True)

if __name__ == "__main__":
    main()
