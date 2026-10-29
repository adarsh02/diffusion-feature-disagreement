"""Shared settings for sampling and filtering."""
from __future__ import annotations

FINAL = {
    "kappa": 0.05,
    "space": "mae",              # Feature model used for these settings.
    "cfg": 1.0,
    "M": 5,
    "top_frac": 0.10,
    "sigma_blur": 1.0,           # Blur size on the model's internal image grid.
    "probe_radius_rel": 0.05,    # Probe distance: 5% of each state's standard deviation.
    "mode": "mae",
    "window_frac": (0.50, 0.62),
    "leaf": "z_t",
    "updates_per_step": 1,
}

# Sampling settings for each image size.
PROTOCOL = {
    128: {"steps": 50, "backend": "adm", "dataset": "imagenet128", "latent": (3, 128, 128)},
    256: {"steps": 50, "backend": "uvit", "dataset": "imagenet256", "latent": (4, 32, 32)},
    512: {"steps": 50, "backend": "uvit", "dataset": "imagenet512", "latent": (4, 64, 64)},
}

# Sampling steps used by the filtering experiment.
FILTER_FRACS = (0.30, 0.40, 0.50, 0.60, 0.80, 0.90)

# Score every step from 40% to 80% of sampling. The six fixed points are also kept.
DENSE_LO, DENSE_HI = 0.40, 0.80
FILTER_SCORES = {"earlier": (0.30, 0.40), "middle": (0.50, 0.60), "late": (0.80, 0.90)}

def window_indices(n_steps: int, frac=None) -> list[int]:
    """Return the sampling-step numbers inside a fractional window."""
    lo, hi = frac or FINAL["window_frac"]
    return [i for i in range(int(n_steps)) if lo <= i / n_steps < hi]

def dense_indices(n_steps: int) -> list[int]:
    """Return every step used for detailed filtering scores."""
    return [i for i in range(int(n_steps)) if DENSE_LO <= i / n_steps < DENSE_HI]

def filter_indices(n_steps: int) -> dict[float, int]:
    """Convert each sampling fraction to a zero-based step number."""
    import math
    return {p: int(math.floor(p * n_steps)) for p in FILTER_FRACS}
