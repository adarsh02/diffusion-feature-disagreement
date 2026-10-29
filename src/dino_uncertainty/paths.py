"""Paths to the large files that are not stored in Git.

Model files, starting states, FID data and real images are not downloaded by this project.
They are read from ``FILES_ROOT``:

    assets/
      models/imagenet{256,512}_uvit_huge.pth, autoencoder_kl_ema.pth, 128x128_diffusion.pt
      starting_points/imagenet{128,256,512}/{X_T.pth,y.pth}
      fid_stats/imagenet{128,256,512}/{m.pt,s.pt}
      imagenet/imagenet{128,256,512}/val/...

Set the ``FILES_ROOT`` environment variable to this folder. No other source repository is needed.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project folders.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = PROJECT_ROOT / "configs"

# Large files kept outside Git.
# ASSETS_ROOT is still accepted for older environment settings.
FILES = Path(os.environ.get("FILES_ROOT", os.environ.get("ASSETS_ROOT", PROJECT_ROOT / "assets")))
ASSETS = FILES  # Keep older scripts that imported this name working.

MODELS = FILES / "models"
STARTING_POINTS = FILES / "starting_points"
FID_STATS = FILES / "fid_stats"
IMAGENET = FILES / "imagenet"

UVIT_CKPT = MODELS / "imagenet256_uvit_huge.pth"
UVIT512_CKPT = MODELS / "imagenet512_uvit_huge.pth"
VAE_CKPT = MODELS / "autoencoder_kl_ema.pth"
ADM128_CKPT = MODELS / "128x128_diffusion.pt"


def starting_points(resolution: int) -> Path:
    """Directory holding the fixed X_T.pth and y.pth for `resolution`.

    Both arms of every comparison read the same file, which is what makes a guided and an
    unguided run differ only by the guidance.
    """
    d = STARTING_POINTS / f"imagenet{resolution}"
    if not d.is_dir():
        raise FileNotFoundError(f"starting points for imagenet{resolution} not at {d}")
    return d


def fid_stats(resolution: int) -> Path:
    """Directory holding the reference Inception statistics m.pt and s.pt."""
    d = FID_STATS / f"imagenet{resolution}"
    if not d.is_dir():
        raise FileNotFoundError(f"FID reference statistics for imagenet{resolution} not at {d}")
    return d


# The real-image folders differ slightly between resolutions.
_VAL_SUBDIR = {64: ("val", "box"), 128: ("val", "box"),
               256: ("val", "box", "all"), 512: ("val", "box", "all")}


def imagenet_val(resolution: int) -> Path:
    """Directory of real ImageNet val images at `resolution`, layout differences absorbed."""
    try:
        parts = _VAL_SUBDIR[resolution]
    except KeyError:
        raise ValueError(
            f"no known layout for imagenet{resolution}; known: {sorted(_VAL_SUBDIR)}"
        ) from None
    d = IMAGENET / f"imagenet{resolution}" / Path(*parts)
    if not d.is_dir():
        raise FileNotFoundError(f"real images for imagenet{resolution} not at {d}")
    return d


# Hugging Face model names. Their downloaded weights are stored in HF_HOME.
DINOV2_REGISTERS = "facebook/dinov2-with-registers-base"   # the encoder the method uses
DINOV2_NO_REGISTERS = "facebook/dinov2-base"               # negative control only
MAE = "facebook/vit-mae-base"

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))


def check_assets(resolution: int | None = None) -> None:
    """Check the required model files before a long run starts."""
    required = [VAE_CKPT]
    if resolution == 512:
        required.append(UVIT512_CKPT)
    elif resolution == 256:
        required.append(UVIT_CKPT)
    elif resolution == 128:
        required = [ADM128_CKPT]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing required files (this project does not download them; see README.md):\n  "
            + "\n  ".join(missing)
        )
    if not HF_HOME.exists():
        raise FileNotFoundError(
            f"HF_HOME={HF_HOME} does not exist. Encoder weights must be fetched on the "
            "LOGIN node before the job runs; compute nodes may have no network."
        )
