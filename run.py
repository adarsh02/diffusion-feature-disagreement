#!/usr/bin/env python
"""Run an experiment from a YAML config: python run.py configs/<name>.yaml"""
from __future__ import annotations
import argparse
import runpy
import shlex
import sys
from pathlib import Path

import yaml



def flags(**kw):
    """Convert keyword arguments to command-line flags, dropping false/None values."""
    out = []
    for k, v in kw.items():
        if v is None or v is False:
            continue
        out.append("--" + k.replace("_", "-"))
        if v is True:
            continue
        out += [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
    return out


def build(cfg):
    """Turn settings from a YAML file into one or more commands."""
    if not isinstance(cfg, dict):
        raise SystemExit("the YAML file must contain names and values")
    if "experiment" not in cfg:
        raise SystemExit("the YAML file must include an 'experiment' setting")
    kind = cfg["experiment"]

    if kind == "unguided":
        return "runners.generate", flags(
            arm=cfg.get("arm", "pool"), resolution=cfg["resolution"], n=cfg["n"],
            batch_size=cfg.get("batch_size"), shard_size=cfg.get("shard_size"),
            M=cfg.get("M"), top_frac=cfg.get("top_frac"),
            probe_radius=cfg.get("probe_radius"), space=cfg.get("encoder"),
            steps=cfg.get("steps"),
            out=cfg["out"])

    if kind == "guidance":
        method = cfg.get("method", "local_probe")
        if method != "local_probe":
            raise SystemExit(f"unknown method {method!r}; this repository implements local_probe")
        # Apply the dose to both limits on the guidance update size.
        d = cfg.get("dose", 1.0)
        argv = flags(probes="local", space=cfg.get("encoder", "dino"),
                     s0_rel=0.05 * d, s0_step_frac=d, cfg=cfg.get("cfg"),
                     resolution=cfg["resolution"], n=cfg["n"], window=cfg["window"],
                     top_frac=cfg.get("top_frac"), M=cfg.get("M"),
                     probe_radius=cfg.get("probe_radius"),
                     batch_size=cfg.get("batch_size"), shard_size=cfg.get("shard_size"),
                     decode_chunk=cfg.get("decode_chunk"), out=cfg["out"],
                     plain_gradient=cfg.get("plain_gradient", False))
        return "runners.generate_local_probe", argv

    if kind == "filtering":
        stage = cfg.get("stage", "all")
        allowed = {"score", "sweep", "retention", "all"}
        if stage not in allowed:
            raise SystemExit(f"unknown filtering stage {stage!r}; choose one of {sorted(allowed)}")
        pool_n = cfg.get("pool_n", 12000)
        pool = Path(cfg.get("pool", str(Path(cfg.get("root", "outputs"))
                                        / f"imagenet{cfg['resolution']}" / "pool")))
        root = cfg.get("root", str(pool.parent.parent))
        expected_pool = Path(root) / f"imagenet{cfg['resolution']}" / "pool"
        if pool.resolve() != expected_pool.resolve():
            raise SystemExit(f"pool must be {expected_pool}; change root to use another output folder")
        steps = []
        if stage in ("score", "all"):
            steps.append(("runners.score_dense", flags(
                resolution=cfg["resolution"], n=pool_n,
                batch_size=cfg.get("batch_size"), shard_size=cfg.get("shard_size"),
                probe_mode=cfg.get("probe_mode"), probe_radius=cfg.get("probe_radius"),
                out=cfg["scores"])))
        if stage in ("sweep", "all"):
            steps.append(("runners.filter_sweep", flags(
                resolution=cfg["resolution"], dense=cfg["scores"], root=root,
                pool_n=pool_n, keep=cfg.get("keep", 10000),
                random_subsets=cfg.get("random_subsets", 20), seed=cfg.get("seed"),
                dense_agg=cfg.get("sweep_agg", "dino_all"),
                range_width=cfg.get("range_width"),
                skip_sparse=cfg.get("skip_sparse", False))))
        if stage in ("retention", "all"):
            for enc in cfg.get("encoders", ["dino"]):
                for rule in cfg.get("patch_rules", ["all"]):
                    steps.append(("runners.filter_retention_curve", flags(
                        resolution=cfg["resolution"], dense=cfg["scores"],
                        dense_agg=f"{enc}_{rule}", window=cfg["window"],
                        root=root, pool_n=pool_n,
                        keeps=cfg.get("keeps"),
                        random_subsets=cfg.get("random_subsets", 20),
                        seed=cfg.get("seed"))))
        return steps

    raise SystemExit(f"unknown experiment: {kind}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args()
    if not a.config.is_file():
        raise SystemExit(f"settings file not found: {a.config}")
    cfg = yaml.safe_load(a.config.read_text())

    # Make the source package available when running directly from the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

    steps = build(cfg)
    if not isinstance(steps, list):
        steps = [steps]
    for mod, argv in steps:
        print("$ " + shlex.join(["python", "-m", mod, *argv]), flush=True)
        if a.dry_run:
            continue
        sys.argv = [mod] + argv
        runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    main()
