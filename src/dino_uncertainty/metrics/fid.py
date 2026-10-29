"""Calculate ImageNet FID features and record which reference files were used."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from dino_uncertainty.paths import fid_stats

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def reference_stats(dataset: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the saved mean and covariance for real images."""
    root = fid_stats(int(str(dataset).replace("imagenet", "")))
    mean_path, cov_path = root / "m.pt", root / "s.pt"
    if not mean_path.is_file() or not cov_path.is_file():
        raise FileNotFoundError(f"missing FID reference statistics under {root}")
    mean = np.asarray(torch.load(mean_path, map_location="cpu"), dtype=np.float64)
    cov = np.asarray(torch.load(cov_path, map_location="cpu"), dtype=np.float64)
    if mean.shape != (2048,) or cov.shape != (2048, 2048):
        raise RuntimeError(f"unexpected {dataset} FID stats: mean {mean.shape}, cov {cov.shape}")
    provenance = {
        "dataset": dataset,
        "split": "ImageNet validation",
        "mean_path": str(mean_path),
        "cov_path": str(cov_path),
        "mean_sha256": _sha256(mean_path),
        "cov_sha256": _sha256(cov_path),
        "inception": "pytorch-fid TF-ported InceptionV3, pool3 2048-d",
        "input": "RGB uint8 converted to [0,1]; model resize_input and normalize_input enabled",
    }
    return mean, cov, provenance

def load_inception(device: torch.device) -> torch.nn.Module:
    """Load the same TF-ported pool3 network used to build the fixed statistics."""
    from pytorch_fid.inception import InceptionV3

    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionV3(
        [block], resize_input=True, normalize_input=True, use_fid_inception=True
    ).to(device).eval()

@torch.no_grad()
def inception_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 50,
) -> torch.Tensor:
    """Return CPU float32 pool3 features for RGB uint8 images."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected uint8 (N,3,H,W), got {images.dtype} {tuple(images.shape)}")
    chunks = []
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size].to(device).float().div_(255.0)
        pred = model(x)[0]
        if pred.shape[-2:] != (1, 1):
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        chunks.append(pred.flatten(1).cpu())
    out = torch.cat(chunks)
    if out.shape != (len(images), 2048) or not torch.isfinite(out).all():
        raise RuntimeError(f"invalid Inception features: {tuple(out.shape)}")
    return out

def fid_from_features(features: torch.Tensor | np.ndarray, dataset: str) -> tuple[float, dict]:
    """Compute FID from one pooled feature matrix, never from shard-level FIDs."""
    from pytorch_fid.fid_score import calculate_frechet_distance

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2048 or len(values) < 2:
        raise ValueError(f"expected (N,2048) features with N>=2, got {values.shape}")
    real_mean, real_cov, provenance = reference_stats(dataset)
    fake_mean = values.mean(0)
    fake_cov = np.cov(values, rowvar=False)
    return float(calculate_frechet_distance(real_mean, real_cov, fake_mean, fake_cov)), provenance
