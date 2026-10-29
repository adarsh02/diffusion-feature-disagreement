"""Image quality and diversity measures based on saved features."""
from __future__ import annotations
import numpy as np
import torch
from scipy import linalg

def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Calculate Frechet distance between two sets of features."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    # Add a small offset if the covariance calculation is unstable.
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"sqrtm produced a large imaginary component: "
                             f"{np.max(np.abs(covmean.imag)):.3e}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))

def kid(a: np.ndarray, b: np.ndarray, subsets: int = 100, subset_size: int = 1000,
        seed: int = 20260830) -> tuple[float, float]:
    """Calculate KID over several random subsets."""
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b), subset_size)
    d = a.shape[1]
    vals = []
    for _ in range(subsets):
        x = a[rng.choice(len(a), n, replace=False)]
        y = b[rng.choice(len(b), n, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum() / (n * (n - 1)) + kyy.sum() / (n * (n - 1))
                    - 2 * kxy.mean())
    v = np.asarray(vals)
    return float(v.mean()), float(v.std())

@torch.no_grad()
def dinov2_features(extractor, images_u8: torch.Tensor, device, chunk: int = 32) -> np.ndarray:
    """Extract one DINOv2 feature vector for each RGB image."""
    out = []
    for i in range(0, images_u8.shape[0], chunk):
        x = images_u8[i:i + chunk].to(device).float() / 127.5 - 1.0
        out.append(extractor.cls_token(x).cpu().numpy())
    return np.concatenate(out).astype(np.float64)

def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3) -> tuple[float, float]:
    """Calculate feature-space precision and recall."""
    import torch as _t
    def _r(x):
        t = _t.from_numpy(x).float()
        d = _t.cdist(t, t)
        d.fill_diagonal_(float("inf"))
        return t, d.topk(k, largest=False).values[:, -1]
    tr, rr = _r(real)
    tf, rf = _r(fake)
    d_fr = _t.cdist(tf, tr)                       # (fake, real)
    precision = float((d_fr <= rr.unsqueeze(0)).any(dim=1).float().mean())
    recall = float((d_fr.T <= rf.unsqueeze(0)).any(dim=1).float().mean())
    return precision, recall

def hf_energy_ratio(guided_u8, unguided_u8, mask=None) -> float:
    """Compare high-frequency detail between guided and unguided samples."""
    import torch as _t, torch.nn.functional as _F
    ker = _t.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    def _e(x):
        g = (x.float() / 255.0).mean(1, keepdim=True)
        lap = _F.conv2d(g, ker, padding=1).pow(2)
        if mask is not None:
            m = _F.interpolate(mask.float(), size=lap.shape[-2:], mode="nearest")
            return float((lap * m).sum() / m.sum().clamp_min(1))
        return float(lap.mean())
    a, b = _e(guided_u8), _e(unguided_u8)
    return a / b if b > 0 else float("nan")

def paired_bootstrap(per_seed: dict, metric_fn, n_boot: int = 100, seed: int = 20260830):
    """Estimate uncertainty by resampling matched random-seed results."""
    rng = np.random.default_rng(seed)
    arms = list(per_seed)
    n = len(next(iter(per_seed.values())))
    vals = {a: [] for a in arms}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for a in arms:
            vals[a].append(metric_fn(per_seed[a][idx]))
    return {a: np.asarray(v) for a, v in vals.items()}
