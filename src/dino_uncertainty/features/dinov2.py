"""Extract DINOv2 features for image patches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dino_uncertainty.features.base import FeatureExtractor, to_unit_interval
from dino_uncertainty.paths import DINOV2_REGISTERS

class DINOv2Features(FeatureExtractor):
    name = "dinov2"

    def __init__(self, model_id: str = DINOV2_REGISTERS, device="cuda", dtype=torch.float32,
                 allow_no_registers: bool = False, input_size: int | None = None,
                 geometry: str = "crop", layer: int | None = None):
        """Load DINOv2. `input_size` is the image size passed to the model."""
        from transformers import AutoImageProcessor, AutoModel

        if geometry not in ("crop", "resize"):
            raise ValueError(f"geometry must be 'crop' or 'resize', got {geometry!r}")

        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.geometry = geometry
        self.layer = layer

        self.model = AutoModel.from_pretrained(model_id).to(self.device, dtype).eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        self.num_registers = int(getattr(cfg, "num_register_tokens", 0) or 0)

        # Register tokens keep the spatial patch features suitable for disagreement maps.
        if self.num_registers == 0 and not allow_no_registers:
            raise ValueError(
                f"{model_id!r} has no register tokens. Use {DINOV2_REGISTERS!r} for "
                "spatial patch disagreement. Set allow_no_registers=True only for a "
                "comparison with the non-register model."
            )

        proc = AutoImageProcessor.from_pretrained(model_id)
        self.patch_size = int(cfg.patch_size)
        self.dim = int(cfg.hidden_size)
        self.num_prefix = 1 + self.num_registers      # CLS + registers, none of them spatial

        if geometry == "crop":
            self.crop = int(input_size or proc.crop_size["height"])
            self.resize_to = int(proc.size["shortest_edge"])
            assert proc.do_center_crop, "processor is expected to centre-crop; geometry assumes it"
            self.input_size = self.crop
            self.grid = self.crop // self.patch_size
            self.stride = float(self.patch_size)
            # For 256px images, only the centre crop changes the image.
            self.offset = float((self.resize_to - self.crop) / 2.0)
        else:
            self.input_size = int(input_size or proc.crop_size["height"])
            if self.input_size % self.patch_size:
                raise ValueError(f"input_size {self.input_size} is not a multiple of the "
                                 f"patch size {self.patch_size}")
            self.crop = self.input_size
            self.resize_to = self.input_size
            self.grid = self.input_size // self.patch_size
            self.offset = 0.0
            self.stride = 256.0 / self.grid       # Input pixels covered by one patch.
        self.name = (f"dinov2@{self.input_size}"
                     + ("" if geometry == "crop" else "_resize")
                     + ("" if layer is None else f"_L{layer}"))

        self._mean = torch.tensor(proc.image_mean, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(proc.image_std, device=self.device, dtype=dtype).view(1, 3, 1, 1)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        x = to_unit_interval(images.to(self.device, self.dtype))
        if self.geometry == "crop":
            if x.shape[-1] != self.resize_to:
                x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bicubic",
                                  align_corners=False, antialias=True).clamp(0, 1)
            o = int(self.offset)
            x = x[:, :, o:o + self.crop, o:o + self.crop]
        else:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        return (x - self._mean) / self._std

    def forward_grad(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch features while keeping gradients for guidance."""
        x = self._preprocess(images)
        if self.layer is None:
            tokens = self.model(pixel_values=x).last_hidden_state
        else:
            tokens = self.model(pixel_values=x, output_hidden_states=True).hidden_states[self.layer]
        return tokens[:, self.num_prefix:, :].float()

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(images)
        if self.layer is None:
            tokens = self.model(pixel_values=x).last_hidden_state
        else:
            hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
            tokens = hs[self.layer]
            # These layer features are normalised later before disagreement is measured.
        patches = tokens[:, self.num_prefix:, :]  # Keep only spatial patch tokens.
        if patches.shape[1] != self.grid ** 2:
            raise ValueError(
                f"got {patches.shape[1]} patch tokens, expected {self.grid**2} "
                f"({self.grid}x{self.grid}) for input {self.input_size} at patch "
                f"{self.patch_size}; total tokens {tokens.shape[1]}, prefix {self.num_prefix}"
            )
        return patches.float()

    @torch.no_grad()
    def cls_token(self, images: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(images)
        if self.layer is None:
            return self.model(pixel_values=x).last_hidden_state[:, 0, :].float()
        return self.model(pixel_values=x, output_hidden_states=True).hidden_states[
            self.layer][:, 0, :].float()

    def describe(self) -> dict:
        d = super().describe()
        d.update(model_id=self.model_id, patch_size=self.patch_size,
                 num_register_tokens=self.num_registers, crop=self.crop,
                 resize_to=self.resize_to, input_size=self.input_size,
                 geometry=self.geometry, layer=self.layer,
                 n_blocks=int(self.model.config.num_hidden_layers))
        return d
