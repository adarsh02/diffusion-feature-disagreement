"""Calculate FID and paired changes for unguided and guided samples."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dino_uncertainty.metrics.fid import fid_from_features, reference_stats
from dino_uncertainty.config import PROTOCOL

LAP = torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
# Image IDs used while choosing the settings. Their overlap with evaluation is reported.
DEV_IDS = {"experiment1": (0, 100), "sweeps_EFG": (100, 150), "confirm": (200, 250),
           "audit": (300, 302), "sweeps_HIJ": (350, 400)}

def hf(u8):
    g = (u8.float() / 255.).mean(1, keepdim=True)
    return F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), LAP).pow(2).flatten(1).mean(1)

def psnr(a, b):
    m = ((a.float() - b.float()) / 255.).pow(2).flatten(1).mean(1)
    return torch.where(m > 0, 10 * torch.log10(1 / m.clamp_min(1e-30)),
                       torch.full_like(m, float("inf")))

def load_shards(d: Path, n: int, want_images=True):
    """Load saved groups in image-ID order."""
    files = sorted((d / "shards").glob("*.pt"))
    if not files:
        return None
    ids, feats, labs, imgs, scores = [], [], [], [], {}
    for f in files:
        r = torch.load(f, map_location="cpu")
        ids.append(torch.as_tensor(r["ids"])); feats.append(r["incep"]); labs.append(r["labels"])
        if want_images:
            imgs.append(r["images"])
        for k, v in (r.get("scores") or {}).items():
            scores.setdefault(k, []).append(v)
    ids = torch.cat(ids)
    o = torch.argsort(ids)
    ordered = bool(torch.equal(o, torch.arange(len(ids))))
    def fix(t):
        return t if ordered else t[o]
    out = {"ids": fix(ids), "incep": fix(torch.cat(feats)), "labels": fix(torch.cat(labs)),
           "scores": {k: fix(torch.cat(v)) for k, v in scores.items()}}
    if want_images:
        out["images"] = fix(torch.cat(imgs))
    if len(out["ids"]) < n:
        print(f"[final_fid] {d}: only {len(out['ids'])}/{n} images present")
    return out

def _spans(d: Path):
    """Read the image-ID range stored in each filename."""
    out = []
    for f in sorted((d / "shards").glob("*.pt")):
        lo, hi = (int(x) for x in f.stem.split("_"))
        out.append((lo, hi, f))
    return out

class ArmImages:
    """Read an image range while keeping one saved group in memory."""

    def __init__(self, d: Path):
        self.spans = _spans(d)
        self._lo = self._hi = -1
        self._img = None

    def __call__(self, lo, hi):
        if not (self._lo <= lo and hi <= self._hi):
            for a, b, f in self.spans:
                if a <= lo < b:
                    r = torch.load(f, map_location="cpu")
                    self._lo, self._hi, self._img = a, b, r["images"]
                    break
            else:
                raise KeyError(f"no shard covers id {lo} in {self.spans and self.spans[0][2].parent}")
        if hi > self._hi:
            raise ValueError(f"range {lo}..{hi} straddles a shard boundary")
        return self._img[lo - self._lo:hi - self._lo]

def blocks(a: "ArmImages", b: "ArmImages", n: int, cap: int = 50):
    """ID ranges that lie inside ONE shard of each arm, chunked to at most `cap` images.
    The two arms may use different shard sizes, so the block edges are the union of both."""
    edges = sorted({0, n} | {x for lo, hi, _ in a.spans for x in (lo, hi)}
                   | {x for lo, hi, _ in b.spans for x in (lo, hi)})
    edges = [e for e in edges if 0 <= e <= n]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        for s in range(lo, hi, cap):
            out.append((s, min(s + cap, hi)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, required=True, choices=[128, 256, 512])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--guided-arm", default="guided",
                    help="folder of guided images to compare with the unguided pool")
    a = ap.parse_args()

    res = a.resolution; ds = PROTOCOL[res]["dataset"]
    root = Path(a.root) / f"imagenet{res}"
    # Keep results for differently named guided groups in separate folders.
    out = root / ("analysis" if a.guided_arm == "guided" else f"analysis_{a.guided_arm}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load features, IDs and labels here, but read RGB images one saved group at
    # a time because keeping all 512x512 images in memory would need about 31 GB.
    pool = load_shards(root / "pool", 12000, want_images=False)
    guided = load_shards(root / a.guided_arm, a.n, want_images=False)
    if pool is None or guided is None:
        raise SystemExit(f"need both pool and {a.guided_arm} shards")

    if len(pool["ids"]) != 12000 or not torch.equal(pool["ids"], torch.arange(12000)):
        raise SystemExit("unguided pool must contain exactly unique IDs 0..11999")
    if len(guided["ids"]) != a.n or not torch.equal(guided["ids"], torch.arange(a.n)):
        raise SystemExit(f"guided arm must contain exactly unique IDs 0..{a.n - 1}")
    n = a.n
    # The first n pool images use the same starting tensors and labels as the
    # guided images.
    ug = {k: (v[:n] if torch.is_tensor(v) else v) for k, v in pool.items() if k != "scores"}
    gd = {k: (v[:n] if torch.is_tensor(v) else v) for k, v in guided.items() if k != "scores"}
    if not torch.equal(ug["ids"], gd["ids"]):
        raise SystemExit("unguided and guided ID sets differ; the pairing would be wrong")
    if not torch.equal(ug["labels"], gd["labels"]):
        raise SystemExit("unguided and guided class labels differ; the pairing would be wrong")
    u_img = ArmImages(root / "pool")
    g_img = ArmImages(root / a.guided_arm)

    _, _, prov = reference_stats(ds)
    fid_u, _ = fid_from_features(ug["incep"].numpy(), ds)
    fid_g, _ = fid_from_features(gd["incep"].numpy(), ds)
    t_fid = time.time() - t0
    print(f"[final_fid] {ds} n={n}: unguided {fid_u:.4f} -> ours {fid_g:.4f} "
          f"({fid_g - fid_u:+.4f})", flush=True)

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lp = None
    if not a.no_lpips:
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LP
            lp = LP(net_type="vgg", normalize=True).to(dev)
            with torch.no_grad():
                k = max(1, min(4, n // 2))
                probe = u_img(0, 2 * k).float().to(dev) / 255.
                s = float(lp(probe[:k], probe[:k]))
                d_ = float(lp(probe[:k], probe[k:2 * k]))
            print(f"[final_fid] LPIPS scale check: identical {s:.6f}  different {d_:.4f}")
        except Exception as exc:
            lp = None; print(f"[final_fid] LPIPS unavailable ({type(exc).__name__}); null")

    t1 = time.time()
    rows = []
    for i, (blo, bhi) in enumerate(blocks(u_img, g_img, n)):
        u, g = u_img(blo, bhi), g_img(blo, bhi)
        p = psnr(g, u)
        ad = (g.float() - u.float()).abs().flatten(1).mean(1)
        hu, hg = hf(u), hf(g)
        lv = [None] * len(u)
        if lp is not None:
            with torch.no_grad():
                uu, gg = u.float().to(dev) / 255., g.float().to(dev) / 255.
                lv = [float(lp(gg[j:j + 1], uu[j:j + 1])) for j in range(len(u))]
        for j in range(len(u)):
            rows.append({"id": int(ug["ids"][blo + j]), "label": int(ug["labels"][blo + j]),
                         "psnr_db": float(p[j]),
                         "identical": bool(torch.isinf(p[j])),
                         "lpips": lv[j], "mean_abs_rgb_255": float(ad[j]),
                         "hf_unguided": float(hu[j]), "hf_guided": float(hg[j]),
                         "hf_ratio": float(hg[j] / max(float(hu[j]), 1e-20))})
        if blo % 2000 < (bhi - blo):
            print(f"[final_fid] paired metrics {blo}/{n}", flush=True)
    t_pair = time.time() - t1

    with open(out / "per_image_guidance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fin = [r["psnr_db"] for r in rows if np.isfinite(r["psnr_db"])]
    lps = [r["lpips"] for r in rows if r["lpips"] is not None]
    dev_overlap = {k: len(set(range(*v)) & set(range(n))) for k, v in DEV_IDS.items()}
    summary = {
        "dataset": ds, "resolution": res, "n": n, "guided_arm": a.guided_arm,
        "fid_unguided_ours": fid_u, "fid_guided_ours": fid_g, "delta_fid": fid_g - fid_u,
        "fid_reference": prov,
        "fid_note": "computed from pooled per-image Inception features over all IDs; "
                    "shard FIDs are never averaged",
        "psnr_db_mean": float(np.mean(fin)) if fin else None,
        "psnr_db_std": float(np.std(fin)) if fin else None,
        "identical_fraction": float(np.mean([r["identical"] for r in rows])),
        "psnr_definition": "10*log10(1/MSE) on native-resolution RGB in [0,1]",
        "psnr_summary": "finite pairs only; identical pairs are represented by identical_fraction",
        "lpips_mean": float(np.mean(lps)) if lps else None,
        "lpips_std": float(np.std(lps)) if lps else None,
        "lpips_input": "native-resolution RGB in [0,1], torchmetrics normalize=True",
        "mean_abs_rgb_255": float(np.mean([r["mean_abs_rgb_255"] for r in rows])),
        "hf_ratio_mean": float(np.mean([r["hf_ratio"] for r in rows])),
        "hf_ratio_dataset": float(np.mean([r["hf_guided"] for r in rows])
                                  / max(np.mean([r["hf_unguided"] for r in rows]), 1e-20)),
        "hf_definition": "mean squared 4-neighbour grayscale Laplacian with replicate padding",
        "paired_metric_caveat": "PSNR/LPIPS/|dRGB|/HF measure how much the image CHANGED "
                                "relative to its own unguided counterpart. None of them is a "
                                "quality measure and none of them supports 'better'.",
        "development_id_overlap": dev_overlap,
        "development_id_note": "IDs inspected while choosing the configuration are INSIDE the "
                               "evaluation set; this set is not held out.",
        "runtime_sec": {"fid": t_fid, "paired_metrics": t_pair},
    }
    (out / "guidance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[final_fid] wrote {out/'guidance_summary.json'}", flush=True)
    print(f"[final_fid] dev-ID overlap with the evaluation set: {dev_overlap}", flush=True)

if __name__ == "__main__":
    main()
