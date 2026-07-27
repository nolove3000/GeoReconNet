import tempfile
import unittest
from pathlib import Path

import torch

from high_resolution_clean.pytorch.model import DamageMetricAccumulator, HighResGeneratorTorch, HybridLossTorch
from high_resolution_clean.pytorch.train import build_lr_scheduler, prune_best_history


class MainModelPyTorchTests(unittest.TestCase):
    def test_best_history_keeps_only_top_three_validation_ious(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for epoch, iou in [(1, 0.1), (2, 0.4), (3, 0.3), (4, 0.2), (5, 0.5)]:
                (directory / f"best_epoch_{epoch:03d}_val_iou_{iou:.6f}.pt").touch()
            removed = prune_best_history(directory, keep=3)
            remaining = sorted(path.name for path in directory.glob("*.pt"))
        self.assertEqual(len(removed), 2)
        self.assertEqual(
            remaining,
            [
                "best_epoch_002_val_iou_0.400000.pt",
                "best_epoch_003_val_iou_0.300000.pt",
                "best_epoch_005_val_iou_0.500000.pt",
            ],
        )

    def test_scheduler_uses_patience_five_and_factor_one_half(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=2e-4)
        scheduler = build_lr_scheduler(optimizer)
        scheduler.step(1.0)
        for _ in range(5):
            scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-4)
        scheduler.step(1.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)

    def test_latent_reshape_preserves_keras_nhwc_flat_order(self):
        model = HighResGeneratorTorch(out_size=128)
        flat = torch.arange(8 * 8 * 256, dtype=torch.float32)[None]
        latent = model._reshape_latent(flat)
        self.assertEqual(tuple(latent.shape), (1, 256, 8, 8))
        for channel, row, column in [(0, 0, 0), (7, 2, 3), (255, 7, 7)]:
            expected_index = (row * 8 + column) * 256 + channel
            self.assertEqual(float(latent[0, channel, row, column]), float(expected_index))

    def test_model_has_no_dropout_or_batch_norm(self):
        model = HighResGeneratorTorch(out_size=128)
        self.assertFalse(any(isinstance(layer, torch.nn.Dropout) for layer in model.modules()))
        self.assertFalse(
            any(
                isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                for layer in model.modules()
            )
        )

    def test_tensorflow_style_initialization_uses_zero_biases(self):
        model = HighResGeneratorTorch(out_size=128)
        parameterized_layers = (
            layer
            for layer in model.modules()
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d))
        )
        for layer in parameterized_layers:
            self.assertTrue(bool(torch.all(layer.bias == 0)))

    def test_full_model_has_expected_trainable_parameter_count(self):
        model = HighResGeneratorTorch(out_size=1024)
        parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertEqual(parameter_count, 18_086_289)

    def test_reduced_resolution_forward_shape_and_probability_range(self):
        model = HighResGeneratorTorch(out_size=128).eval()
        with torch.no_grad():
            prediction = model(torch.zeros((1, 132), dtype=torch.float32))
        self.assertEqual(tuple(prediction.shape), (1, 1, 128, 128))
        self.assertTrue(bool(torch.all((prediction >= 0) & (prediction <= 1))))

    def test_hybrid_loss_is_finite(self):
        probability = torch.full((2, 1, 8, 8), 0.5)
        solid_truth = torch.ones_like(probability)
        loss = HybridLossTorch()(probability, solid_truth)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_damage_metrics_use_zero_as_the_damage_class(self):
        metrics = DamageMetricAccumulator()
        solid_probability = torch.tensor([[[[0.1, 0.9], [0.1, 0.9]]]])
        solid_truth = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        metrics.update(solid_probability, solid_truth)
        result = metrics.compute()
        self.assertAlmostEqual(result["damage_iou"], 1 / 3)
        self.assertAlmostEqual(result["damage_dice"], 0.5)
        self.assertAlmostEqual(result["damage_precision"], 0.5)
        self.assertAlmostEqual(result["damage_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
