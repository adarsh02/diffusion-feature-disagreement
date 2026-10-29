"""Shared functions for patch scores, probe noise and guidance."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def masked_values(U, mask, grid, state_hw):
    """Average selected patch scores for each image."""
    Ul = F.interpolate(U.reshape(-1, 1, grid, grid), size=state_hw,
                       mode="bilinear", align_corners=False)
    numer = (mask * Ul).flatten(1).sum(1)
    denom = mask.flatten(1).sum(1).clamp_min(1.0)
    return numer / denom


def to_latent(U, grid, hw):
    """Resize patch scores to the latent grid."""
    return F.interpolate(U.reshape(-1, 1, grid, grid), size=hw,
                         mode="bilinear", align_corners=False)


def build_mask_latent(Ul, top_frac):
    """Select the highest-scoring fraction of grid cells."""
    B = Ul.shape[0]
    k = max(1, int(math.ceil(top_frac * Ul[0, 0].numel())))
    flat = Ul.reshape(B, -1).detach()
    idx = torch.argsort(flat, dim=1, descending=True, stable=True)[:, :k]
    m = torch.zeros_like(flat).scatter_(1, idx, 1.0)
    return m.view_as(Ul), int(k)


def probe_noise(ids, timestep, shape, M, device, stream=0):
    """Create repeatable probe noise for each image and sampling step.

    Using the image ID means that changing the batch size or restarting a run does not change
    the noise for an image.
    """
    out = torch.empty((M, len(ids), *shape))
    for b, gid in enumerate(ids):
        for m in range(M):
            seed = (int(gid) * 1000003 + int(timestep) * 10007
                    + int(stream) * 1000000007 + m) % (2**31 - 1)
            out[m, b] = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    return out.to(device)


def eps_cfg(bundle, x, t, y, cfg=1.0, grad=False):
    """Predict noise, with optional classifier-free guidance."""
    f = bundle.eps_grad if grad else bundle.eps
    if float(cfg) == 1.0:
        return f(x, t, y)
    if not math.isfinite(float(cfg)):
        raise ValueError("classifier-free guidance strength must be finite")
    if not hasattr(bundle, "null_y"):
        raise ValueError("this backend has no unconditional class; cfg must be 1.0")
    ec = f(x, t, y)
    eu = f(x, t, bundle.null_y(x.shape[0]))
    return eu + float(cfg) * (ec - eu)
