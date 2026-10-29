"""Load and run the ImageNet-128 ADM model."""
from __future__ import annotations
from dataclasses import dataclass

import torch

from dino_uncertainty.paths import ADM128_CKPT
EPS_CHANNELS = 3          # The other three output channels describe predicted variance.
IMAGE_SIZE = 128

class IdentityDecoder:
    """ADM works directly with images, so no VAE conversion is needed."""

    @staticmethod
    def decode(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def encode(x: torch.Tensor) -> torch.Tensor:
        return x

@dataclass
class ADMBundle:
    unet: torch.nn.Module
    device: torch.device
    vae: IdentityDecoder = IdentityDecoder()

    def eps_grad(self, x_t: torch.Tensor, t: int, y: torch.Tensor) -> torch.Tensor:
        """Predict noise and keep gradients with respect to the input image."""
        t_tensor = torch.full((x_t.shape[0],), int(t), device=x_t.device, dtype=torch.long)
        out = self.unet(x_t, t_tensor, y=y)
        if out.shape[1] != 2 * EPS_CHANNELS:
            raise RuntimeError(
                f"expected {2 * EPS_CHANNELS} output channels from the ADM UNet "
                f"(epsilon + learned variance), got {out.shape[1]}")
        return out[:, :EPS_CHANNELS]

    @torch.no_grad()
    def eps(self, x_t: torch.Tensor, t: int, y: torch.Tensor) -> torch.Tensor:
        return self.eps_grad(x_t, t, y)

    @torch.no_grad()
    def decode(self, z: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        return z

def disable_attention_checkpointing(unet: torch.nn.Module) -> int:
    """Turn off memory checkpointing because guidance needs input gradients."""
    import types
    from dino_uncertainty.model_code.guided_diffusion.unet_openai import AttentionBlock

    patched = 0
    for m in unet.modules():
        if isinstance(m, AttentionBlock):
            m.forward = types.MethodType(lambda self, x: self._forward(x), m)
            patched += 1
    if patched == 0:
        raise RuntimeError("no AttentionBlock found in the ADM UNet; the guidance gradient "
                           "would fail inside gradient checkpointing")
    return patched

def load_adm_bundle(device: str | torch.device = "cuda", dtype=torch.float32) -> ADMBundle:
    """Load the published ImageNet-128 ADM model."""
    from dino_uncertainty.model_code.guided_diffusion.unet_openai import UNetModel

    if not ADM128_CKPT.is_file():
        raise FileNotFoundError(
            f"ADM model file not found: {ADM128_CKPT}. Set FILES_ROOT as described in README.md."
        )
    device = torch.device(device)
    unet = UNetModel(
        image_size=128,
        in_channels=3,
        model_channels=256,
        out_channels=6,
        num_res_blocks=2,
        attention_resolutions=(4, 8, 16),
        dropout=0.0,
        channel_mult=(1, 1, 2, 3, 4),
        num_classes=1000,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=4,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    unet.load_state_dict(torch.load(ADM128_CKPT, map_location="cpu"))
    unet.to(device=device, dtype=dtype).eval().requires_grad_(False)
    n = disable_attention_checkpointing(unet)
    print(f"[adm] attention checkpointing disabled on {n} blocks (input gradients require it)")
    return ADMBundle(unet=unet, device=device)

def load_adm_scheduler():
    """Build the noise schedule used by the ImageNet-128 model."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    return DDPMScheduler(
        beta_start=0.0001,
        beta_end=0.02,
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
