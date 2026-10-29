"""Raw pixels pooled to a matching grid -- the control space."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dino_uncertainty.features.base import FeatureExtractor, to_unit_interval

class PixelFeatures(FeatureExtractor):
    name = "pixel"

    def __init__(self, grid: int = 16, pool: str = "patch", device="cuda",
                 image_size: int = 256, patch: int = 16):
        if pool not in ("patch", "mean"):
            raise ValueError(f"unknown pool {pool!r}")
        if grid < 1 or patch < 1:
            raise ValueError(f"grid and patch must be >= 1, got {grid}, {patch}")
        self.device = torch.device(device)
        self.image_size = image_size
        self.grid = int(grid)
        self.pool = pool
        self.patch_size = int(patch)
        self.resize_to = self.grid * self.patch_size      # exact by construction
        self.dim = 3 * self.patch_size ** 2 if pool == "patch" else 3
        self.offset = 0.0
        self.stride = image_size / self.grid              # source px per token
        self.name = f"pixel{self.grid}" if pool == "patch" else f"pixel{self.grid}_mean"

    def forward_grad(self, images: torch.Tensor) -> torch.Tensor:
        """Grad-capable twin of __call__ (which is @torch.no_grad)."""
        x = to_unit_interval(images.to(self.device))
        if x.shape[-1] != self.resize_to:
            x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        B, C, _, _ = x.shape
        g, pz = self.grid, self.patch_size
        x = x.reshape(B, C, g, pz, g, pz).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C, pz, pz)
        if self.pool == "mean":
            return x.mean(dim=(-1, -2)).float()
        return x.reshape(B, g * g, self.dim).float()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = to_unit_interval(images.to(self.device).float())
        if x.shape[-1] != self.resize_to:
            x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        B, C = x.shape[0], x.shape[1]
        p, g = self.patch_size, self.grid
        # (B, C, g, p, g, p) -> tokens in row-major (i, j) order, matching every other space
        x = x.reshape(B, C, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C, p, p)
        if self.pool == "mean":
            return x.mean(dim=(-1, -2))
        return x.reshape(B, g * g, self.dim)

    def describe(self) -> dict:
        d = super().describe()
        d.update(pool=self.pool, patch_size=self.patch_size, resize_to=self.resize_to)
        return d
