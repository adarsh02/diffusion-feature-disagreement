"""Build the requested feature extractor."""

from __future__ import annotations

from dino_uncertainty.features.base import FeatureExtractor, common_valid_mask
from dino_uncertainty.features.dinov2 import DINOv2Features
from dino_uncertainty.features.mae import MAEFeatures
from dino_uncertainty.features.pixel import PixelFeatures
from dino_uncertainty.features.untrained import UntrainedViTFeatures

__all__ = ["FeatureExtractor", "DINOv2Features", "MAEFeatures", "PixelFeatures",
           "UntrainedViTFeatures", "build_extractor", "common_valid_mask", "SPEC_ALIASES"]

# Short names used by the comparison scripts.
SPEC_ALIASES = {
    # Models using the same 14x14 patch grid.
    "pixel14":      {"name": "pixel", "grid": 14},
    "dinov2@196":   {"name": "dinov2", "input_size": 196, "geometry": "resize"},
    "mae":          {"name": "mae"},
    # Other DINOv2 sizes.
    "dinov2@518":   {"name": "dinov2", "input_size": 518, "geometry": "resize"},
    "dinov2@224":   {"name": "dinov2", "input_size": 224, "geometry": "resize"},
    "dinov2crop":   {"name": "dinov2", "geometry": "crop"},
    # Pixel comparisons at the same grid sizes.
    "pixel16":      {"name": "pixel", "grid": 16},
    "pixel32":      {"name": "pixel", "grid": 32},
    # Intermediate DINOv2 layers.
    "dinov2@196_L9": {"name": "dinov2", "input_size": 196, "geometry": "resize", "layer": 9},
    "dinov2@196_L6": {"name": "dinov2", "input_size": 196, "geometry": "resize", "layer": 6},
    "dinov2@196_L11": {"name": "dinov2", "input_size": 196, "geometry": "resize", "layer": 11},
    "dinov2@392": {"name": "dinov2", "input_size": 392, "geometry": "resize"},
    # Untrained comparison model.
    "untrained@196": {"name": "untrained_vit", "input_size": 196, "geometry": "resize"},
}

def build_extractor(spec: str | dict, device="cuda", seeds=None) -> FeatureExtractor:
    """Build a feature extractor from a short name or settings dictionary."""
    if isinstance(spec, str):
        spec = SPEC_ALIASES.get(spec, {"name": spec})
    spec = dict(spec)
    name = spec.pop("name")

    if name == "dinov2":
        return DINOv2Features(device=device, **spec)
    if name == "mae":
        return MAEFeatures(device=device, **spec)
    if name == "pixel":
        return PixelFeatures(device=device, pool=spec.pop("pool", "patch"), **spec)
    if name == "pixel_mean":
        return PixelFeatures(device=device, pool="mean", **spec)
    if name == "untrained_vit":
        seed = spec.pop("seed", None)
        if seed is None:
            seed = getattr(seeds, "encoder_init", 777) if seeds is not None else 777
        return UntrainedViTFeatures(device=device, seed=seed, **spec)
    raise ValueError(f"unknown feature space {name!r}")
