"""Guide sampling by reducing disagreement between nearby predictions.

The update is:

    z <- z - a * g / ||g||,   a = min(s0_rel * ||z||, s0_step_frac * ||dz_sampler||)

Here, `g` is the score gradient and `a` limits the size of the change. Gradients are calculated
one probe at a time to reduce memory use. An optional line search makes the change smaller when
the first attempt does not lower disagreement.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from dino_uncertainty.guidance.core import (masked_values, to_latent, build_mask_latent,
                                            eps_cfg, probe_noise)

CHUNK = {"probes": 1}   # Number of probes handled together. One matches the reported runs.

PROBE_STREAM = 4471          # Probes used to check an update.
HELDOUT_STREAM = 4472        # Extra probes used only for reporting.
CTRL_STREAM = 4473           # Random directions used by the control experiment.

def parse_indices(spec, steps):
    """Read either a step range such as '40-45' or a list such as '40,43,45'."""
    spec = str(spec)
    if "," in spec:
        idx = sorted({int(v) for v in spec.split(",") if v.strip() != ""})
    else:
        lo, hi = (int(v) for v in spec.split("-"))
        idx = list(range(lo, hi + 1))
    if not idx or min(idx) < 0 or max(idx) >= steps:
        raise SystemExit(f"window {spec} outside 0..{steps - 1}")
    return idx

def probe_spread(mode, delta, sig, scale):
    """Return the distance between the centre and each probe."""
    return delta if mode == "local" else scale * sig

def probe_states(z, eta_m, mode, delta, sig, eps0, scale=1.0):
    """Build one nearby state while keeping gradients with respect to `z`."""
    if mode == "local":
        return z + delta * eta_m
    return z - sig * eps0 + scale * sig * eta_m   # A*x0hat + s*sig*xi, x0hat linear in z

def probe_features(bundle, ex, z, eta_m, y, t, A, sig, mode, delta, eps0, cfg=1.0,
                   grad=False, scale=1.0):
    x = probe_states(z, eta_m, mode, delta, sig, eps0, scale)
    b = (x - sig * eps_cfg(bundle, x, t, y, cfg, grad=grad)) / A
    d = bundle.vae.decode(b) if grad else bundle.decode(b)
    return F.normalize(ex.forward_grad(d) if grad else ex(d), dim=-1)

@torch.no_grad()
def probe_variance(bundle, ex, z, eta, y, t, A, sig, mode, delta, eps0, M, cfg=1.0, scale=1.0):
    """Return per-patch probe variance and mean features without storing gradients."""
    k = CHUNK["probes"]
    if k <= 1:
        f = torch.stack([probe_features(bundle, ex, z, eta[m], y, t, A, sig, mode, delta, eps0,
                                        cfg, scale=scale) for m in range(M)])
    else:
        # Several independent probes can be handled in one model call to save time.
        B = z.shape[0]
        outs = []
        for lo in range(0, M, k):
            hi = min(lo + k, M); n = hi - lo
            ss = scale.repeat(n, 1, 1, 1) if torch.is_tensor(scale) else scale
            fa = probe_features(bundle, ex, z.repeat(n, 1, 1, 1),
                                torch.cat([eta[m] for m in range(lo, hi)], 0),
                                y.repeat(n), t, A, sig, mode,
                                delta.repeat(n, 1, 1, 1), eps0.repeat(n, 1, 1, 1),
                                cfg, scale=ss)
            outs.extend(fa[i * B:(i + 1) * B] for i in range(n))
        f = torch.stack(outs)
    return (f - f.mean(0)).pow(2).sum(-1).mean(0), f.mean(0)

def exact_grad(bundle, ex, z, eta, y, t, A, sig, mode, delta, eps0, M, mask, fbar, cfg=1.0, scale=1.0):
    """Calculate the score gradient one probe at a time to save memory."""
    g = torch.zeros_like(z)
    hw = z.shape[-2:]
    for m in range(M):
        zl = z.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            f = probe_features(bundle, ex, zl, eta[m], y, t, A, sig, mode, delta, eps0,
                               cfg, grad=True, scale=scale)
            Ui = (f - fbar).pow(2).sum(-1) / M
            Li = masked_values(Ui, mask, ex.grid, hw).sum()
        gm, = torch.autograd.grad(Li, zl)
        g = g + gm.detach()
    return g

def joint_graph_grad(bundle, ex, z, eta, y, t, A, sig, mode, delta, eps0, M, mask, cfg=1.0, scale=1.0):
    """Calculate all probe gradients together for small comparison runs."""
    zl = z.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        f = torch.stack([probe_features(bundle, ex, zl, eta[m], y, t, A, sig, mode, delta,
                                        eps0, cfg, grad=True, scale=scale) for m in range(M)])
        U = (f - f.mean(0)).pow(2).sum(-1).mean(0)       # fbar NOT detached
        L = masked_values(U, mask, ex.grid, z.shape[-2:]).sum()
    g, = torch.autograd.grad(L, zl)
    return g.detach()

def _unit(v):
    return v / v.flatten(1).norm(dim=1).clamp_min(1e-20).view(-1, 1, 1, 1)

def _rand_dir(z, ids, t):
    out = torch.empty_like(z)
    for b, gid in enumerate(ids):
        seed = (int(gid) * 1000003 + int(t) * 10007 + CTRL_STREAM * 1000000007) % (2 ** 31 - 1)
        out[b] = torch.randn(z.shape[1:], generator=torch.Generator().manual_seed(seed)).to(z.device)
    return _unit(out)

def step_length(z_norm, sampler_step_norm, s0_rel, s0_step_frac):
    """Limit the update using both the current state and the unguided sampler step."""
    return torch.minimum(s0_rel * z_norm, s0_step_frac * sampler_step_norm)


def descend(z, direction, a):
    """Move distance `a` in the chosen direction."""
    return z + a.view(-1, 1, 1, 1) * direction


def guided_step(bundle, ex, sch, z, y, ids, t, i, *, mode, top_frac, M, s0_rel, s0_step_frac,
                max_halvings, all_tol, e0, control=False, replay=None, cfg=1.0, accept_test=True,
                probe_stream=None, probe_radius=0.05, probe_scale=1.0, match_spread=False,
                max_doublings=0, heldout=True):
    """Run one guided sampling step.

    When `accept_test` is true, a change is kept only if it lowers disagreement.
    """
    B = z.shape[0]
    ac = sch.sched.alphas_cumprod.to(z.device)
    al = ac[t]; A, sig = al.sqrt(), (1 - al).sqrt()
    hw = z.shape[-2:]
    dev = z.device

    ps = PROBE_STREAM if probe_stream is None else int(probe_stream)
    eta = probe_noise(ids, t, tuple(z.shape[1:]), M, dev, stream=ps)
    eta_h = (probe_noise(ids, t, tuple(z.shape[1:]), M, dev, stream=HELDOUT_STREAM)
             if heldout else None)
    # Keep the probe distance fixed during this guided step.
    delta = (probe_radius * z.detach().flatten(1).std(dim=1)).view(-1, 1, 1, 1)
    scale = float(probe_scale)
    if match_spread:
        # Give both probe types the same spread for the comparison experiment.
        if mode == "tweedie":
            # Each image has its own local probe distance.
            scale = delta / sig
        else:  # Use the Tweedie probe distance.
            delta = torch.full_like(delta, float(sig))
    _scale_mean = float(scale.flatten().mean()) if torch.is_tensor(scale) else float(scale)
    spread = probe_spread(mode, float(delta.flatten().mean()), float(sig), _scale_mean)

    U0, fbar = probe_variance(bundle, ex, z, eta, y, t, A, sig, mode, delta, e0, M, cfg,
                              scale=scale)
    mask, cells = build_mask_latent(to_latent(U0, ex.grid, hw), top_frac)
    L0 = masked_values(U0, mask, ex.grid, hw)
    A0 = U0.mean(1)
    if heldout:
        U0h, _ = probe_variance(bundle, ex, z, eta_h, y, t, A, sig, mode, delta, e0, M, cfg,
                                scale=scale)
        L0h, A0h = masked_values(U0h, mask, ex.grid, hw), U0h.mean(1)
    else:
        # These extra probes are reported but do not affect the update.
        U0h = None
        L0h = A0h = torch.full((B,), float("nan"), device=dev)

    # Compare the update with the size of an unguided sampling step.
    with torch.no_grad():
        ord_step = (sch.sched.step(e0, t, z).prev_sample - z).flatten(1).norm(dim=1)
    zn = z.flatten(1).norm(dim=1)
    s0 = step_length(zn, ord_step, s0_rel, s0_step_frac)

    if control:
        d = _rand_dir(z, ids, t)
        gnorm = float("nan")
    else:
        g = exact_grad(bundle, ex, z, eta, y, t, A, sig, mode, delta, e0, M, mask, fbar,
                           cfg, scale=scale)
        gnorm = float(g.flatten(1).norm(dim=1).mean())
        d = -_unit(g)

    # Start with the full allowed update. The optional checks below may make it smaller.
    tried = torch.zeros(B, dtype=torch.long, device=dev)
    acc = torch.ones(B, dtype=torch.bool, device=dev)
    s_acc = s0.clone()

    if replay is not None:
        # Use the same update sizes and skipped images as the matched comparison run.
        s_acc = torch.tensor([replay.get(f"{int(gid)}_{i}", 0.0) for gid in ids],
                             device=dev, dtype=z.dtype)
        acc = s_acc > 0
        tried = acc.long()
    elif accept_test:
        # Keep halving the update until it lowers disagreement; otherwise skip it.
        acc = torch.zeros(B, dtype=torch.bool, device=dev)
        s_acc = torch.zeros(B, device=dev)
        for k in range(max_halvings + 1):
            live = ~acc
            if not bool(live.any()):
                break
            s = (s0 / (2 ** k))
            zt = z + (s * live.to(z.dtype)).view(-1, 1, 1, 1) * d
            U1, _ = probe_variance(bundle, ex, zt, eta, y, t, A, sig, mode, delta, e0, M, cfg,
                                   scale=scale)
            ok = live & (masked_values(U1, mask, ex.grid, hw) < L0) & (U1.mean(1) <= A0 * (1 + all_tol))
            tried = tried + live.long()
            s_acc = torch.where(ok, s, s_acc)
            acc = acc | ok

    # Some comparison runs also try larger updates after the first one succeeds.
    doublings = torch.zeros(B, dtype=torch.long, device=dev)
    if max_doublings > 0 and replay is None and bool(acc.any()):
        grow = acc.clone()
        for _ in range(max_doublings):
            if not bool(grow.any()):
                break
            s = s_acc * 2
            zt = z + (s * grow.to(z.dtype)).view(-1, 1, 1, 1) * d
            U1, _ = probe_variance(bundle, ex, zt, eta, y, t, A, sig, mode, delta, e0, M, cfg,
                                   scale=scale)
            ok = grow & (masked_values(U1, mask, ex.grid, hw) < L0) & (U1.mean(1) <= A0 * (1 + all_tol))
            tried = tried + grow.long()
            doublings = doublings + ok.long()
            s_acc = torch.where(ok, s, s_acc)
            grow = ok

    z_new = descend(z, d, s_acc * acc.to(z.dtype))
    if bool(acc.any()):
        U2, _ = probe_variance(bundle, ex, z_new, eta, y, t, A, sig, mode, delta, e0, M,
                               cfg, scale=scale)
        U2h = (probe_variance(bundle, ex, z_new, eta_h, y, t, A, sig, mode, delta, e0, M,
                              cfg, scale=scale)[0] if heldout else None)
        with torch.no_grad():
            # Recalculate epsilon at the changed diffusion state.
            e_new = eps_cfg(bundle, z_new, t, y, cfg)
    else:
        U2, U2h, e_new = U0, U0h, e0
    L1, A1 = masked_values(U2, mask, ex.grid, hw), U2.mean(1)
    L1h, A1h = ((masked_values(U2h, mask, ex.grid, hw), U2h.mean(1)) if U2h is not None
                else (L0h, A0h))

    rec = {"step": i, "t": t, "ids": [int(v) for v in ids], "cells": cells,
           "grad_norm": gnorm, "control": bool(control), "replayed": replay is not None,
           "sigma_t": float(sig), "alpha_bar_t": float(al), "probe_radius": float(probe_radius),
           "probe_scale": _scale_mean, "probe_spread": spread,
           "delta_mean": float(delta.flatten().mean()),
           "U0_masked_mean": float(L0.mean()), "U0_all_mean": float(A0.mean()),
           "s0": [float(v) for v in s0], "ord_step_norm": [float(v) for v in ord_step],
           "z_norm": [float(v) for v in zn],
           "trials": [int(v) for v in tried], "accepted": [bool(v) for v in acc],
           "doublings": [int(v) for v in doublings],
           "s_accepted": [float(v) for v in s_acc],
           "masked_var_before": [float(v) for v in L0], "masked_var_after": [float(v) for v in L1],
           "all_var_before": [float(v) for v in A0], "all_var_after": [float(v) for v in A1],
           "heldout_masked_before": [float(v) for v in L0h],
           "heldout_masked_after": [float(v) for v in L1h],
           "heldout_all_before": [float(v) for v in A0h],
           "heldout_all_after": [float(v) for v in A1h],
           "rel_edit": [float(v) for v in ((z_new - z).flatten(1).norm(dim=1) / zn.clamp_min(1e-12))]}
    return z_new, e_new, rec
