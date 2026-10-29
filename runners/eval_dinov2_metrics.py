"""Compare generated and real images using DINOv2 features."""
from __future__ import annotations
import argparse, json, os, time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dino_uncertainty.config import PROTOCOL
from dino_uncertainty.metrics.features import frechet_distance, kid, dinov2_features, precision_recall
from dino_uncertainty.paths import PROJECT_ROOT, imagenet_val
from dino_uncertainty.provenance import atomic_json, fingerprint_files, sha256_file
from runners.eval_fid import _spans


def feature_cache_settings(files, extractor, n, resolution):
    """Tie cached features to the actual input files and encoder implementation."""
    code = [__file__, PROJECT_ROOT / "src/dino_uncertainty/features/dinov2.py",
            PROJECT_ROOT / "src/dino_uncertainty/features/base.py",
            PROJECT_ROOT / "src/dino_uncertainty/metrics/features.py"]
    return {"n": n, "resolution": resolution, "encoder": extractor.describe(),
            "encoder_revision": getattr(extractor.model.config, "_commit_hash", None),
            "torch_version": str(torch.__version__),
            "transformers_version": version("transformers"), "pillow_version": version("pillow"),
            "inputs_sha256": fingerprint_files(files),
            "feature_code_sha256": fingerprint_files(code)}


def cached_features(path, settings, compute, expected_shape):
    """Reuse a cache only when its inputs, settings and stored array still match."""
    path = Path(path)
    settings = json.loads(json.dumps(settings))
    metadata = path.with_suffix(".json")

    def valid(values):
        return (isinstance(values, np.ndarray) and values.shape == expected_shape
                and np.issubdtype(values.dtype, np.floating)
                and np.isfinite(values).all())

    if path.is_file() and metadata.is_file():
        try:
            saved = json.loads(metadata.read_text())
            if (isinstance(saved, dict) and saved.get("settings") == settings
                    and saved.get("sha256") == sha256_file(path)):
                values = np.load(path, allow_pickle=False)
                if valid(values):
                    print(f"[fd-dino] reuse {path.name}", flush=True)
                    return values
        except (OSError, ValueError, TypeError):
            pass

    values = np.asarray(compute())
    if not valid(values):
        raise RuntimeError(f"invalid features for {path}: expected finite {expected_shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)
    atomic_json({"settings": settings, "sha256": sha256_file(path)}, metadata)
    return values


def arm_features(extractor, d: Path, n: int, dev, chunk: int):
    """Load DINOv2 features while keeping one saved group in memory."""
    out, seen = [], 0
    for lo, hi, f in _spans(d):
        if lo >= n:
            break
        rec = torch.load(f, map_location="cpu")
        if lo != seen or not torch.equal(torch.as_tensor(rec["ids"]), torch.arange(lo, hi)):
            raise SystemExit(f"{d}: missing, duplicated or out-of-order image IDs at {f.name}")
        img = rec["images"][: max(0, min(hi, n) - lo)]
        out.append(dinov2_features(extractor, img, dev, chunk=chunk))
        seen += len(img)
        del rec, img
    if seen != n:
        raise SystemExit(f"{d}: found {seen} images, need {n}")
    return np.concatenate(out)

def real_features(extractor, n: int, res: int, dev, chunk: int, batch: int = 200, files=None):
    """Extract features from real validation images at the requested size."""
    files = sorted(imagenet_val(res).glob("*.png"))[:n] if files is None else files
    if len(files) < n:
        raise SystemExit(f"only {len(files)} val PNGs, need {n}")
    out = []
    for i in range(0, n, batch):
        arrs = []
        for f in files[i:i + batch]:
            im = Image.open(f).convert("RGB")
            if im.size != (res, res):
                im = im.resize((res, res), Image.BICUBIC)
            arrs.append(np.asarray(im, dtype=np.uint8))
        u8 = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()
        out.append(dinov2_features(extractor, u8, dev, chunk=chunk))
    return np.concatenate(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[128, 256, 512])
    ap.add_argument("--n", type=int, default=10000, help="images from each sample group")
    ap.add_argument("--ref-n", type=int, default=None,
                    help="number of real images; defaults to --n")
    ap.add_argument("--arms", nargs="+", default=["pool", "guided"],
                    help="sample folders; simple names are read below outputs/imagenet<size>")
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--tag", default="",
                    help="optional name added to the output filename")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--pr-subset", type=int, default=5000,
                    help="number of real and generated images used for precision/recall")
    a = ap.parse_args()

    res, n = a.resolution, a.n
    ref_n = a.ref_n if a.ref_n is not None else n
    if min(n, ref_n, a.pr_subset) < 4 or a.chunk < 1:
        ap.error("n, ref-n and pr-subset must be at least 4; chunk must be positive")
    root = Path(a.root) / f"imagenet{res}"
    out = root / "analysis"; out.mkdir(parents=True, exist_ok=True)
    cache = out / "fd_dinov2_cache"; cache.mkdir(exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    from dino_uncertainty.features import build_extractor
    ex = build_extractor("dinov2crop", device=dev)
    desc = ex.describe()
    if not desc.get("num_register_tokens"):
        raise SystemExit(f"DINOv2 must be the register variant, got {desc.get('model_id')}")
    print(f"[fd-dino] {desc['model_id']} registers={desc['num_register_tokens']} "
          f"dim={desc.get('dim')}", flush=True)

    t0 = time.time()
    rp = cache / f"real_n{ref_n}.npy"
    real_files = sorted(imagenet_val(res).glob("*.png"))[:ref_n]
    real = cached_features(
        rp, feature_cache_settings(real_files, ex, ref_n, res),
        lambda: real_features(ex, ref_n, res, dev, a.chunk, files=real_files),
        (ref_n, ex.dim))
    print(f"[fd-dino] real {real.shape} in {time.time()-t0:.0f}s", flush=True)

    feats = {}
    for arm in a.arms:
        arm_path = Path(arm)
        src = arm_path if arm_path.is_absolute() or len(arm_path.parts) > 1 else root / arm
        name = arm.replace("/", "__").replace("\\", "__").replace(":", "_")
        p = cache / f"{name}_n{n}.npy"
        t1 = time.time()
        inputs = [f for lo, hi, f in _spans(src) if lo < n]
        feats[arm] = cached_features(
            p, feature_cache_settings(inputs, ex, n, res),
            lambda: arm_features(ex, src, n, dev, a.chunk), (n, ex.dim))
        print(f"[fd-dino] {arm} {feats[arm].shape} in {time.time()-t1:.0f}s", flush=True)

    rows = {}
    for arm, f in feats.items():
        fd = frechet_distance(f, real)
        km, ks = kid(f, real, subsets=50, subset_size=min(1000, n))
        k = a.pr_subset
        pr, rc = precision_recall(real[:k], f[:k])
        rows[arm] = {"fd_dinov2": fd, "kid_mean": km, "kid_std": ks,
                     "precision": pr, "recall": rc, "n": n}
        print(f"[fd-dino] {arm:10s} FD {fd:9.4f}  KID {km:.6f}+/-{ks:.6f}  "
              f"P {pr:.4f}  R {rc:.4f}", flush=True)

    base = "pool" if "pool" in rows else a.arms[0]
    summary = {
        "dataset": PROTOCOL[res]["dataset"], "resolution": res, "n": n,
        "encoder": desc, "baseline_arm": base, "arms": rows,
        "deltas_vs_baseline": {
            arm: {"d_fd_dinov2": r["fd_dinov2"] - rows[base]["fd_dinov2"],
                  "d_kid": r["kid_mean"] - rows[base]["kid_mean"],
                  "d_precision": r["precision"] - rows[base]["precision"],
                  "d_recall": r["recall"] - rows[base]["recall"]}
            for arm, r in rows.items() if arm != base},
        "reference": f"real ImageNet val PNGs resized to {res}px, first {ref_n}",
        "arm_n": n, "ref_n": ref_n,
        "guidance_note": "If guidance uses DINOv2, these metrics share its feature encoder "
                         "and are not an independent evaluation. Report Inception FID "
                         "alongside them.",
        "absolute_caveat": "absolute FD is not comparable across resolutions -- the reference is "
                           "resized per resolution. Only the delta between arms at ONE "
                           "resolution is claimed.",
        "runtime_sec": time.time() - t0,
    }
    (out / f"fd_dinov2{('_' + a.tag) if a.tag else ''}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[fd-dino] wrote {out}/fd_dinov2{('_' + a.tag) if a.tag else ''}.json", flush=True)
    for arm, d in summary["deltas_vs_baseline"].items():
        print(f"[fd-dino] {arm} vs {base}: dFD {d['d_fd_dinov2']:+.4f}  "
              f"dKID {d['d_kid']:+.6f}  dP {d['d_precision']:+.4f}  dR {d['d_recall']:+.4f}",
              flush=True)

if __name__ == "__main__":
    main()
