# Patch-Feature Disagreement for Diffusion Sample Filtering and Guidance

## Problem

Diffusion models can generate convincing images, but individual samples may still contain
distorted shapes, repeated object parts, or inconsistent detail. Metrics such as FID
summarise the quality of a collection of images, so a favourable overall score can hide
problems in individual outputs.

This project studies whether instability in intermediate diffusion predictions can help
assess sample quality. The idea is that if small changes to a noisy state produce different
predictions for the same image region, the model may be uncertain about the structure
developing there.

## Approach

At selected sampling steps, small Gaussian perturbations create several nearby versions
of the current diffusion state. The denoiser predicts a clean image from each version.
For latent diffusion models, these predictions are decoded into RGB before being processed
by a frozen DINOv2 or MAE encoder.

The method compares normalised feature vectors at corresponding patch locations across
these predictions. Their variance produces a spatial disagreement map: higher values
indicate regions whose predicted features change more under the perturbations.

This signal is used in two ways:

- **Filtering:** average disagreement across image regions and selected sampling steps
  to obtain one score per sample, then retain completed images with the lowest scores.
- **Guidance:** use the gradient of disagreement to adjust the current diffusion state
  during sampling. Updates have bounded sizes and are checked before acceptance; an
  unsuccessful update is reduced or skipped.

The diffusion model and feature encoder remain frozen throughout. The current implementation
uses 50-step DDIM sampling with ADM at 128 pixels and U-ViT at 256 and 512 pixels.

## Code structure

```text
run.py                          Run experiments from YAML files
configs/                        Experiment settings
runners/                        Generation, filtering, evaluation and diagnostics
src/dino_uncertainty/
  guidance/                     Disagreement scores and guidance updates
  features/                     DINOv2, MAE and control feature extractors
  backends/                     ADM and U-ViT model loading
  metrics/                      FID and feature-space metrics
  model_code/                   Model definitions and third-party licence notices
  config.py                     Shared sampling settings
  paths.py                      Checkpoint and dataset paths
  provenance.py                 Saved run settings and source checks
requirements.txt                Python dependencies
```

## Experiment commands

Run from the repository root with the required checkpoints, starting points and reference
data under `assets/`, or set `FILES_ROOT` to their location. See `paths.py` for the layout.
Outputs are saved under `outputs/imagenet<resolution>/`.

```bash
# Unguided pool: ImageNet-256, 12,000 samples
python run.py configs/unguided_pool.yaml

# Filtering: score, sweep five-step windows and compare retention levels
python run.py configs/filtering_imagenet256.yaml

# DINOv2 guidance at each resolution
python run.py configs/guidance_imagenet128_dinov2.yaml
python run.py configs/guidance_imagenet256_dinov2.yaml
python run.py configs/guidance_imagenet512_dinov2.yaml

# Later guidance window at 256 pixels
python run.py configs/guidance_imagenet256_dinov2_late_window.yaml

# Guidance without candidate checks
python run.py configs/guidance_imagenet256_plain_gradient.yaml
```

For an unguided pool at 128 or 512 pixels, copy `unguided_pool.yaml` and change `resolution`
and `out`. Use a pool at the same resolution as the guided run. Add `--dry-run` to any
command above to inspect it without generating images.

Evaluation commands below use Bash's `PYTHONPATH=src` prefix. In PowerShell, set
`$env:PYTHONPATH = "$PWD\src"` and run the commands without that prefix.

```bash
# FID and paired image comparisons
PYTHONPATH=src python -m runners.eval_fid --resolution 256 --n 10000 --root outputs --guided-arm guided_dino_w30-35_d2.0

# DINOv2 distribution metrics
PYTHONPATH=src python -m runners.eval_dinov2_metrics --resolution 256 --n 10000 --root outputs --arms pool guided_dino_w30-35_d2.0
```

## Results

Reported DINOv2 ImageNet results; lower FID is better. The report and current configs
differ in sampler and guidance settings, so exact reproduction needs the original run records.

| Resolution | Filtering: random subset / retained | Guidance: unguided / guided |
| --- | --- | --- |
| 128 | 10.6055 / **9.5121** | 10.6086 / **9.2724** |
| 256 | 7.9147 / **6.8827** | 7.8918 / **7.1724** |
| 512 | 15.0367 / **12.1618** | 15.0314 / **12.3249** |

Filtering retains 10,000 of 12,000 images using the best tested five-step scoring window;
the random baseline averages 20 equal-sized subsets. Guidance compares 10,000 matched
samples with the same starting noise and class labels.
