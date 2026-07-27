import unittest

import torch
from torch import nn

from ablations.architecture.model import ArchitectureAblationGenerator


class ArchitectureAblationTests(unittest.TestCase):
    def test_single_projection_changes_only_embedding_depth(self):
        model = ArchitectureAblationGenerator("single_projection")
        self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in model.modules()), 1)
        self.assertEqual(model.base_channels, 256)
        self.assertEqual(sum(isinstance(layer, nn.ConvTranspose2d) for layer in model.modules()), 7)
        self.assertEqual(model(torch.zeros(1, 132)).shape, (1, 1, 1024, 1024))

    def test_light_decoder_retains_three_linear_layers(self):
        model = ArchitectureAblationGenerator("light_decoder")
        self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in model.modules()), 3)
        self.assertEqual(model.base_channels, 128)
        self.assertEqual(sum(isinstance(layer, nn.ConvTranspose2d) for layer in model.modules()), 7)
        self.assertEqual(model(torch.zeros(1, 132)).shape, (1, 1, 1024, 1024))

    def test_no_normalization_or_dropout(self):
        for variant in ("single_projection", "light_decoder"):
            model = ArchitectureAblationGenerator(variant)
            self.assertFalse(any(isinstance(layer, (nn.modules.batchnorm._BatchNorm, nn.Dropout)) for layer in model.modules()))


if __name__ == "__main__":
    unittest.main()
