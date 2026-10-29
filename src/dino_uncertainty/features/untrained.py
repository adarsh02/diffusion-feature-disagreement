"""Extract features with an untrained ViT for a comparison run."""

from __future__ import annotations

import torch

from dino_uncertainty.features.base import FeatureExtractor, to_unit_interval
from dino_uncertainty.features.dinov2 import DINOv2Features
from dino_uncertainty.paths import DINOV2_REGISTERS

class UntrainedViTFeatures(DINOv2Features):
    name = "untrained_vit"

    def __init__(self, model_id: str = DINOV2_REGISTERS, device="cuda", dtype=torch.float32,
                 seed: int = 777, input_size: int | None = None, geometry: str = "crop"):
        from transformers import AutoConfig, AutoModel

        # Build the same model shape without loading trained weights.
        cfg = AutoConfig.from_pretrained(model_id)
        prev = torch.get_rng_state()
        try:
            torch.manual_seed(int(seed))  # The previous random state is restored below.
            model = AutoModel.from_config(cfg)
        finally:
            torch.set_rng_state(prev)

        # Reuse the same image preparation, then replace the trained weights.
        super().__init__(model_id=model_id, device=device, dtype=dtype,
                         allow_no_registers=False, input_size=input_size, geometry=geometry)
        self.model = model.to(self.device, self.dtype).eval()
        self.model.requires_grad_(False)
        self.seed = int(seed)
        self.name = "untrained_vit"

    def describe(self) -> dict:
        d = super().describe()
        d.update(name="untrained_vit", pretrained=False, init_seed=self.seed,
                 architecture_of=self.model_id)
        return d
