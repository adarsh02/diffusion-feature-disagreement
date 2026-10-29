"""Record which source files were used for a run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[str | Path]) -> str:
    """Identify an ordered set of inputs by their paths and file contents."""
    digest = hashlib.sha256()
    for item in paths:
        path = Path(item).resolve()
        digest.update(str(path).encode("utf-8") + b"\0")
        digest.update(sha256_file(path).encode("ascii") + b"\0")
    return digest.hexdigest()

def atomic_json(payload: dict, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)

def write_run_settings(destination: str | Path, settings: dict) -> None:
    """Save settings, refusing to mix different runs in the same output folder."""
    destination = Path(destination)
    settings = json.loads(json.dumps(settings))
    if destination.is_file():
        previous = json.loads(destination.read_text())
        changed = [key for key, value in settings.items()
                   if key not in ("gpu", "slurm_job_id") and previous.get(key) != value]
        if changed:
            raise RuntimeError(f"run settings changed ({', '.join(changed)}); "
                               f"use a new output folder instead of {destination.parent}")
    atomic_json(settings, destination)

def source_manifest(paths: Iterable[str | Path], root: str | Path = ".") -> dict:
    """Record each source file and a value used to detect changes."""
    root = Path(root).resolve()
    files = {}
    for item in paths:
        path = Path(item).resolve()
        try:
            name = str(path.relative_to(root))
        except ValueError:
            name = str(path)
        files[name] = {"path": str(path), "sha256": sha256_file(path)}
    return {"root": str(root), "files": files}

def write_or_verify_manifest(destination: str | Path, manifest: dict) -> None:
    """Do not continue a saved run after its source files have changed."""
    destination = Path(destination)
    if destination.is_file():
        previous = json.loads(destination.read_text())
        if previous != manifest:
            raise RuntimeError(
                f"source files changed since {destination} was created; use a new output "
                "folder instead of combining files from different code versions"
            )
        return
    atomic_json(manifest, destination)
