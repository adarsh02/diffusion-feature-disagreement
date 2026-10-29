"""Create a side-by-side figure of matched guided and unguided images."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

def load_arm(path: Path, ids=None, n=None):
    """Load selected saved images from a folder or a single .pt file."""
    if path.suffix == ".pt":
        t = torch.load(path, map_location="cpu")
        t = t["images"] if isinstance(t, dict) else t
        return t if ids is None else t[list(ids)]
    shards = sorted((path / "shards").glob("*.pt"))
    if not shards:
        raise SystemExit(f"no shards under {path}")
    if ids is None:
        ids = list(range(n))
    want, out = set(int(i) for i in ids), {}
    for f in shards:
        lo, hi = (int(x) for x in f.stem.split("_"))
        if not any(lo <= i < hi for i in want):
            continue
        r = torch.load(f, map_location="cpu")
        rid = torch.as_tensor(r["ids"]).tolist()
        for j, g in enumerate(rid):
            if g in want:
                out[g] = r["images"][j].clone()
        del r
        if len(out) == len(want):
            break
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{path}: missing ids {sorted(missing)[:5]}")
    return torch.stack([out[int(i)] for i in ids])

def grid(u8_rows, row_labels, ids, per_image, tile, title, sub, amp):
    k = len(ids)
    pad, head, cap, lab = 5, 40, 15, 96
    W = lab + k * (tile + pad) + pad
    H = head + len(u8_rows) * (tile + cap + pad) + pad
    c = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(c)
    d.text((pad, 7), title, fill=(0, 0, 0))
    d.text((pad, 22), sub, fill=(95, 95, 95))
    for r, (row, rl) in enumerate(zip(u8_rows, row_labels)):
        y = head + r * (tile + cap + pad)
        d.text((pad, y + tile // 2 - 4), rl, fill=(0, 0, 0))
        for i in range(k):
            im = row[i].permute(1, 2, 0).numpy()
            t = Image.fromarray(im).resize((tile, tile), Image.LANCZOS)
            x = lab + i * (tile + pad)
            c.paste(t, (x, y))
            if r == 0:
                d.text((x, y - 11), f"#{ids[i]}", fill=(0, 0, 0))
            if r == len(u8_rows) - 1:
                d.text((x, y + tile + 2), f"{per_image[i]:.2f}/255", fill=(110, 110, 110))
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unguided", required=True, help="folder of saved images or a .pt file")
    ap.add_argument("--guided", required=True, help="folder of saved images or a .pt file")
    ap.add_argument("--pool-n", type=int, default=2000,
                    help="how many leading IDs to search when --select change")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--select", default="change", choices=["change", "first"])
    ap.add_argument("--amp", type=float, default=0,
                    help="difference amplification; 0 = auto (scale the max to 200)")
    ap.add_argument("--tile", type=int, default=132)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    up, gp = Path(a.unguided), Path(a.guided)
    flat = up.suffix == ".pt"
    n = a.pool_n
    if flat:
        U, G = load_arm(up), load_arm(gp)
        n = min(len(U), len(G))
        U, G = U[:n], G[:n]
        ids = list(range(n))
    else:
        ids = list(range(n))
        U, G = load_arm(up, ids), load_arm(gp, ids)

    diff = (G.float() - U.float()).abs()
    per = diff.flatten(1).mean(1)
    order = torch.argsort(per, descending=True) if a.select == "change" else torch.arange(len(per))
    pick = order[:a.k].tolist()
    ids_p = [ids[i] for i in pick]
    Us, Gs, Ds = U[pick], G[pick], diff[pick]

    # Use the 99.5th percentile so a few unusual pixels do not set the scale.
    p995 = float(np.percentile(Ds.numpy(), 99.5))
    amp = a.amp or max(1.0, 200.0 / max(p995, 1e-6))
    Dv = (Ds * amp).clamp(0, 255).to(torch.uint8)

    sel = ("the 8 images guidance moved MOST" if a.select == "change"
           else "the first 8 IDs (unselected)")
    sub = (f"same starting tensor per column | {sel} | difference amplified {amp:.0f}x | "
           f"mean |dRGB| over {n} images: {per.mean():.3f}/255, max {per.max():.2f}")
    img = grid([Us, Gs, Dv], ["unguided", "guided", f"|diff| x{amp:.0f}"],
               ids_p, [float(x) for x in per[pick]], a.tile,
               a.title or f"{up.name} vs {gp.name}", sub, amp)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    stats = {"unguided": str(up), "guided": str(gp), "n_compared": int(n),
             "select": a.select, "ids": ids_p, "amplification": amp,
             "mean_abs_rgb_255": float(per.mean()), "max_abs_rgb_255": float(per.max()),
             "per_image_shown": [float(x) for x in per[pick]],
             "frac_pixels_changed": float((diff > 0.5).float().mean()),
             "png": str(out)}
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[pairs] {n} images | mean |dRGB| {per.mean():.4f}/255 | max {per.max():.2f} | "
          f"{stats['frac_pixels_changed']*100:.1f}% of pixels moved | amp {amp:.0f}x -> {out}",
          flush=True)

if __name__ == "__main__":
    main()
