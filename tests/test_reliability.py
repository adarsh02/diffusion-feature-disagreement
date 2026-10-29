"""Small CPU checks for saved-output reuse and conditioning consistency."""
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from runners.eval_dinov2_metrics import cached_features, feature_cache_settings
from runners.score_dense import validate_shard
import runners.generate_local_probe as generate
from dino_uncertainty.guidance.core import eps_cfg


class CacheTests(unittest.TestCase):
    def test_reuses_only_matching_inputs_and_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "image.txt"
            source.write_text("1")
            cache = Path(directory) / "features.npy"
            extractor = SimpleNamespace(
                model=SimpleNamespace(config=SimpleNamespace(_commit_hash="revision-1")),
                describe=lambda: {"model_id": "test-encoder", "grid": 2})
            calls = []

            def compute():
                calls.append(1)
                return np.full((4, 3), float(source.read_text()))

            def evaluate():
                settings = feature_cache_settings([source], extractor, 4, 256)
                return cached_features(cache, settings, compute, (4, 3))

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue((evaluate() == 1).all())
                evaluate()
                self.assertEqual(len(calls), 1)
                # Even a replacement with the same size and timestamps must invalidate it.
                stat = source.stat()
                source.write_text("2")
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertTrue((evaluate() == 2).all())
                self.assertEqual(len(calls), 2)
                extractor.model.config._commit_hash = "revision-2"
                evaluate()
                self.assertEqual(len(calls), 3)
                extractor.describe = lambda: {"model_id": "test-encoder", "grid": 4}
                evaluate()
                self.assertEqual(len(calls), 4)
                cache.write_bytes(b"interrupted array write")
                evaluate()
                self.assertEqual(len(calls), 5)
                cache.with_suffix(".json").write_text("invalid metadata")
                evaluate()
                self.assertEqual(len(calls), 6)
                cache.with_suffix(".json").unlink()
                evaluate()
                self.assertEqual(len(calls), 7)

    def test_rejects_invalid_new_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npy"
            for values in [np.zeros((2, 3)), np.full((4, 3), np.nan)]:
                with self.subTest(shape=values.shape), self.assertRaises(RuntimeError):
                    cached_features(path, {}, lambda: values, (4, 3))
            self.assertFalse(path.exists())


class ScoreTests(unittest.TestCase):
    def test_accepts_complete_scores_and_rejects_invalid_shards(self):
        labels = torch.tensor([5, 6, 7, 8])
        fractions = {20: 0.4, 21: 0.42}
        record = {"ids": [0, 1, 2, 3], "labels": labels, "fractions": fractions}
        for name in ("mae_top10", "mae_all", "dino_top10", "dino_all"):
            record[name] = {step: torch.ones(4) for step in fractions}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000000_000004.pt"
            torch.save(record, path)
            validate_shard(path, 0, 4, labels, fractions)
            corruptions = []
            for field, value in [("ids", [0, 1, 1, 3]), ("labels", labels + 1),
                                 ("fractions", {20: 0.4})]:
                item = copy.deepcopy(record)
                item[field] = value
                corruptions.append(item)
            for values in [torch.ones(3), torch.full((4,), float("nan")), -torch.ones(4)]:
                item = copy.deepcopy(record)
                item["dino_all"][20] = values
                corruptions.append(item)
            item = copy.deepcopy(record)
            del item["mae_top10"][21]
            corruptions.append(item)
            for index, item in enumerate(corruptions):
                with self.subTest(corruption=index), self.assertRaises(RuntimeError):
                    torch.save(item, path)
                    validate_shard(path, 0, 4, labels, fractions)


class ToyBundle:
    def null_y(self, n):
        return torch.full((n,), 1000)

    def eps(self, x, t, y):
        value = torch.where(y == 1000, 2.0, 4.0).view(-1, 1, 1, 1)
        return torch.ones_like(x) * value

    eps_grad = eps

    def decode(self, z, chunk=8):
        return torch.zeros(len(z), 3, 256, 256)


class ConditioningTests(unittest.TestCase):
    def test_unsupported_backend_and_nonfinite_strength_fail_clearly(self):
        x, y = torch.zeros(2, 4, 2, 2), torch.zeros(2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "unconditional"):
            eps_cfg(SimpleNamespace(eps=lambda *args: x), x, 0, y, 2.0)
        for value in (float("nan"), float("inf")):
            with self.subTest(cfg=value), self.assertRaisesRegex(ValueError, "finite"):
                eps_cfg(ToyBundle(), x, 0, y, value)

    def test_sampling_uses_same_conditioning_on_every_step(self):
        for cfg, expected in [(1.0, 4.0), (2.0, 6.0)]:
            with self.subTest(cfg=cfg), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                torch.save(torch.zeros(4, 4, 2, 2), directory / "X_T.pth")
                torch.save(torch.arange(4), directory / "y.pth")
                predictions, guided_calls = [], []

                def step(epsilon, t, z):
                    predictions.append(epsilon.clone())
                    return SimpleNamespace(prev_sample=z - epsilon)

                def guide(bundle, ex, sch, z, y, ids, t, i, **kwargs):
                    guided_calls.append(kwargs["e0"].clone())
                    return z, kwargs["e0"], {"rel_edit": [0.] * len(ids),
                                              "accepted": [False] * len(ids)}

                scheduler = SimpleNamespace(timesteps=[2, 1, 0], step=step,
                                            alphas_cumprod=torch.full((3,), 0.5))
                args = ["generate", "--resolution", "256", "--n", "4", "--batch-size", "2",
                        "--shard-size", "4", "--window", "1-1", "--s0-rel", ".1",
                        "--s0-step-frac", "1", "--cfg", str(cfg), "--device", "cpu",
                        "--out", str(directory / "output")]
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(sys, "argv", args))
                    stack.enter_context(patch.object(generate, "PROTOCOL", {
                        256: {"steps": 3, "dataset": "imagenet256", "latent": (4, 2, 2)}}))
                    stack.enter_context(patch.object(generate, "load_backend", return_value=(ToyBundle(), scheduler)))
                    stack.enter_context(patch.object(generate, "starting_points", return_value=directory))
                    stack.enter_context(patch.object(generate, "guided_step", side_effect=guide))
                    stack.enter_context(patch("dino_uncertainty.features.build_extractor", return_value=SimpleNamespace(describe=lambda: {})))
                    stack.enter_context(patch("dino_uncertainty.metrics.fid.load_inception", return_value=None))
                    stack.enter_context(patch("dino_uncertainty.metrics.fid.inception_features", side_effect=lambda model, images, device: torch.zeros(len(images), 2048)))
                    stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    generate.main()
                self.assertEqual(len(predictions), 6)
                self.assertEqual(len(guided_calls), 2)
                self.assertTrue(all(torch.allclose(p, torch.full_like(p, expected))
                                    for p in predictions + guided_calls))


if __name__ == "__main__":
    unittest.main()
