import unittest

import torch
from torch import nn

from baselines.simple_deconv.model import SimpleDeconvBaseline


class SimpleDeconvBaselineTests(unittest.TestCase):
    def test_native_full_resolution_output(self):
        model = SimpleDeconvBaseline(latent_channels=128)
        with torch.no_grad():
            output = model(torch.randn(1, 132))

        self.assertEqual(tuple(output.shape), (1, 1, 1024, 1024))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))

    def test_architecture_has_one_dense_projection_and_plain_deconvolutions(self):
        model = SimpleDeconvBaseline()

        self.assertEqual(sum(isinstance(module, nn.Linear) for module in model.modules()), 1)
        self.assertEqual(
            sum(isinstance(module, nn.ConvTranspose2d) for module in model.modules()), 7
        )
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))
        self.assertFalse(any(isinstance(module, nn.Dropout) for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
