"""Load and run U-ViT with its image decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dino_uncertainty.paths import UVIT_CKPT, UVIT512_CKPT, VAE_CKPT


LATENT_CHANNELS = 4
LATENT_SIZE = 32
IMAGE_SIZE = 256
SCALE_FACTOR = 0.18215
NULL_CLASS = 1000  # U-ViT ImageNet-256 has num_classes=1001; 1000 is the CFG null token

@dataclass
class UViTBundle:
    uvit: torch.nn.Module
    vae: torch.nn.Module
    device: torch.device

    # Noise prediction.
    def eps_grad(self, x_t: torch.Tensor, t: int, y: torch.Tensor) -> torch.Tensor:
        """Predict noise while keeping input gradients for guidance."""
        t_tensor = torch.full((x_t.shape[0],), int(t), device=x_t.device, dtype=torch.long)
        out = self.uvit(x_t, t_tensor, y)
        assert out.shape == x_t.shape, f"eps shape {tuple(out.shape)} != x_t {tuple(x_t.shape)}"
        return out

    @torch.no_grad()
    def eps(self, x_t: torch.Tensor, t: int, y: torch.Tensor) -> torch.Tensor:
        """Predict noise without storing gradients."""
        return self.eps_grad(x_t, t, y)

    # Conversion between the latent state and an RGB image.
    @torch.no_grad()
    def decode(self, z: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        """Decode latent states to RGB in small groups to reduce memory use."""
        if z.shape[0] <= chunk:
            return self.vae.decode(z)
        return torch.cat([self.vae.decode(z[i:i + chunk]) for i in range(0, z.shape[0], chunk)],
                         dim=0)

    @torch.no_grad()
    def encode_deterministic(self, x: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        """Encode RGB images to latent states using the posterior mean."""
        outs = []
        for i in range(0, x.shape[0], chunk):
            moments = self.vae.encode_moments(x[i:i + chunk])
            mean, _logvar = torch.chunk(moments, 2, dim=1)
            outs.append(SCALE_FACTOR * mean)
        return torch.cat(outs, dim=0)

    def null_y(self, n: int) -> torch.Tensor:
        return torch.full((n,), NULL_CLASS, device=self.device, dtype=torch.long)

# Model sizes used by the published 256px and 512px checkpoints.
UVIT_GEOMETRY = {256: {"patch_size": 2, "latent": 32}, 512: {"patch_size": 4, "latent": 64}}

def load_uvit_bundle(device: str | torch.device = "cuda", dtype=torch.float32,
                     image_size: int = 256) -> UViTBundle:
    from dino_uncertainty.model_code.uvit.autoencoder import get_model as get_autoencoder
    from dino_uncertainty.model_code.uvit.uvit import UViT

    if image_size not in UVIT_GEOMETRY:
        raise ValueError(f"no U-ViT geometry for image_size={image_size}; "
                         "128 is ADM, not U-ViT")
    geo = UVIT_GEOMETRY[image_size]
    ckpt = UVIT_CKPT if image_size == 256 else UVIT512_CKPT
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"U-ViT model file not found: {ckpt}. Set FILES_ROOT as described in README.md."
        )
    if not VAE_CKPT.is_file():
        raise FileNotFoundError(
            f"VAE model file not found: {VAE_CKPT}. Set FILES_ROOT as described in README.md."
        )

    device = torch.device(device)
    uvit = UViT(img_size=geo["latent"], patch_size=geo["patch_size"], in_chans=LATENT_CHANNELS,
                embed_dim=1152, depth=28, num_heads=16, num_classes=1001, conv=False)
    uvit.load_state_dict(torch.load(ckpt, map_location="cpu"))
    uvit.to(device=device, dtype=dtype).eval().requires_grad_(False)

    vae = get_autoencoder(str(VAE_CKPT), scale_factor=SCALE_FACTOR)
    vae.to(device=device, dtype=dtype).eval().requires_grad_(False)

    return UViTBundle(uvit=uvit, vae=vae, device=device)


def load_uvit_scheduler():
    """Build the noise schedule used by the published U-ViT model."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    return DDPMScheduler.from_config({
        "_class_name": "PNDMScheduler",
        "_diffusers_version": "0.6.0",
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "num_train_timesteps": 1000,
        "set_alpha_to_one": False,
        "skip_prk_steps": True,
        "steps_offset": 1,
        "trained_betas": None,
        "clip_sample": False,
    })
