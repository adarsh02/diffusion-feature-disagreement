"""Common interface for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SIZE = 256

class FeatureExtractor(ABC):
    name: str = "base"
    grid: int = 0
    dim: int = 0

    #: Patch offset and spacing in the 256-pixel image.
    offset: float = 0.0
    stride: float = 0.0

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, 256, 256) in [-1, 1] -> (B, grid*grid, dim)."""

    def cls_token(self, images: torch.Tensor):
        """(B, dim) global token, or None if this space has no such thing."""
        return None

    # Patch locations in the input image.
    def pixel_boxes(self) -> np.ndarray:
        """Return each patch box as (r0, r1, c0, c1) in the input image."""
        boxes = np.zeros((self.grid, self.grid, 4), dtype=np.float32)
        for i in range(self.grid):
            for j in range(self.grid):
                boxes[i, j] = (self.offset + i * self.stride, self.offset + (i + 1) * self.stride,
                               self.offset + j * self.stride, self.offset + (j + 1) * self.stride)
        return boxes

    def coverage_box(self) -> tuple[float, float, float, float]:
        """Return the part of the input image covered by patch features."""
        lo = self.offset
        hi = self.offset + self.grid * self.stride
        return (lo, hi, lo, hi)

    def to_pixel_map(self, grid_map: torch.Tensor, size: int = IMAGE_SIZE) -> torch.Tensor:
        """(B, grid, grid) -> (B, size, size), placed at this space's true offset."""
        assert grid_map.dim() == 3 and grid_map.shape[-1] == self.grid, \
            f"expected (B, {self.grid}, {self.grid}), got {tuple(grid_map.shape)}"
        B = grid_map.shape[0]
        r0, r1, _, _ = self.coverage_box()
        span = int(round(r1 - r0))
        up = F.interpolate(grid_map.unsqueeze(1).float(), size=(span, span),
                           mode="bilinear", align_corners=False).squeeze(1)
        out = torch.full((B, size, size), float("nan"), dtype=torch.float32, device=grid_map.device)
        a, b = int(round(r0)), int(round(r0)) + span
        out[:, a:b, a:b] = up
        return out

    def valid_mask(self, size: int = IMAGE_SIZE) -> torch.Tensor:
        """Return True where the feature extractor covers the image."""
        r0, r1, _, _ = self.coverage_box()
        m = torch.zeros((size, size), dtype=torch.bool)
        m[int(round(r0)):int(round(r1)), int(round(r0)):int(round(r1))] = True
        return m

    def describe(self) -> dict:
        return {"name": self.name, "grid": self.grid, "dim": self.dim,
                "patch_offset_px": self.offset, "patch_stride_px": self.stride,
                "coverage_box_px": list(self.coverage_box())}

def to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], clamped.  Every encoder's preprocessing starts here."""
    return (images.clamp(-1, 1) + 1.0) / 2.0

def common_valid_mask(extractors, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Return the image area covered by every feature extractor."""
    m = torch.ones((size, size), dtype=torch.bool)
    for e in extractors:
        m &= e.valid_mask(size)
    return m
