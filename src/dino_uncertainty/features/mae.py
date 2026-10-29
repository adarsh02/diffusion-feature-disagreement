"""Extract MAE patch features in image order."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dino_uncertainty.features.base import FeatureExtractor, to_unit_interval
from dino_uncertainty.paths import MAE

class MAEFeatures(FeatureExtractor):
    name = "mae"

    def __init__(self, model_id: str = MAE, device="cuda", dtype=torch.float32):
        from transformers import AutoImageProcessor, ViTMAEModel

        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype

        # A zero mask ratio keeps every image patch.
        self.model = ViTMAEModel.from_pretrained(model_id, mask_ratio=0.0)
        self.model.to(self.device, dtype).eval().requires_grad_(False)
        assert float(self.model.config.mask_ratio) == 0.0, "mask_ratio must be 0: we mask nothing"

        proc = AutoImageProcessor.from_pretrained(model_id)
        self.input_size = int(proc.size["height"])
        assert not getattr(proc, "do_center_crop", False), \
            "MAE processor is expected NOT to crop; geometry assumes a plain resize"
        self.patch_size = int(self.model.config.patch_size)
        self.grid = self.input_size // self.patch_size
        self.dim = int(self.model.config.hidden_size)
        self.offset = 0.0
        self.stride = 256.0 / self.grid          # Input pixels covered by one patch.

        self._mean = torch.tensor(proc.image_mean, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(proc.image_std, device=self.device, dtype=dtype).view(1, 3, 1, 1)

    def _identity_noise(self, batch: int) -> torch.Tensor:
        """Choose fixed values so MAE returns patches in the usual image order."""
        L = self.grid ** 2
        base = torch.arange(L, device=self.device, dtype=torch.float32) / float(L)
        return base.unsqueeze(0).expand(batch, L).contiguous()

    def _ordered_patches(self, output) -> torch.Tensor:
        patches = output.last_hidden_state[:, 1:, :]
        if patches.shape[1] != self.grid ** 2:
            raise ValueError(
                f"got {patches.shape[1]} patch tokens, expected {self.grid**2}; mask_ratio "
                f"is {self.model.config.mask_ratio}"
            )
        ids_restore = getattr(output, "ids_restore", None)
        if ids_restore is not None:
            expected = torch.arange(self.grid ** 2, device=ids_restore.device)
            if not torch.equal(ids_restore[0], expected):
                raise RuntimeError("MAE patch tokens are not in spatial raster order")
        return patches.float()

    def forward_grad(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch features while keeping gradients for guidance."""
        x = to_unit_interval(images.to(self.device, self.dtype))
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        x = (x - self._mean) / self._std
        out = self.model(pixel_values=x, noise=self._identity_noise(x.shape[0]))
        return self._ordered_patches(out)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = to_unit_interval(images.to(self.device, self.dtype))
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        x = (x - self._mean) / self._std

        out = self.model(pixel_values=x, noise=self._identity_noise(x.shape[0]))
        return self._ordered_patches(out)

    @torch.no_grad()
    def cls_token(self, images: torch.Tensor) -> torch.Tensor:
        x = to_unit_interval(images.to(self.device, self.dtype))
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        x = (x - self._mean) / self._std
        out = self.model(pixel_values=x, noise=self._identity_noise(x.shape[0]))
        return out.last_hidden_state[:, 0, :].float()

    def describe(self) -> dict:
        d = super().describe()
        d.update(model_id=self.model_id, patch_size=self.patch_size,
                 input_size=self.input_size, mask_ratio=float(self.model.config.mask_ratio),
                 ordering="explicit increasing noise -> identity permutation",
                 resize="full frame to 224x224; bicubic; align_corners=False; antialias=True",
                 colour_order="RGB",
                 input_range="decoded [-1,1] clamped and mapped to [0,1]",
                 normalisation={"mean": self._mean.flatten().detach().cpu().tolist(),
                                "std": self._std.flatten().detach().cpu().tolist()},
                 measurement_grid="14x14 final patch tokens, CLS excluded",
                 resolution_note=("224px measurement upsamples 128px outputs and downsamples "
                                  "256px/512px outputs; sensitivity to fine detail is not "
                                  "resolution invariant"))
        return d
