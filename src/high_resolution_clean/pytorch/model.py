import torch
from torch import nn


TORCH_INITIALIZATION_VERSION = "glorot_uniform_zero_bias_v1"
TORCH_ARCHITECTURE_VERSION = "nhwc_latent_no_norm_no_dropout_v3"


class HighResGeneratorTorch(nn.Module):
    """PyTorch counterpart of ``main_model.HighResGenerator``.

    The layer widths, 8x8 latent grid, transposed-convolution schedule, Swish/SiLU
    activations, and dropout rate match the TensorFlow high-resolution model.
    Solid-class probabilities are returned, matching the TensorFlow final sigmoid.
    """

    def __init__(self, input_dimension=132, out_size=1024, base_channels=256, dropout=0.0):
        super().__init__()
        if out_size not in (128, 256, 512, 1024):
            raise ValueError("out_size must be 128, 256, 512, or 1024")
        self.input_dimension = input_dimension
        self.out_size = out_size
        self.base_channels = base_channels
        self.dropout = dropout
        if self.dropout != 0.0:
            raise ValueError("this architecture requires dropout=0")
        self.dense1 = nn.Linear(input_dimension, 512)
        self.dense2 = nn.Linear(512, 1024)
        self.dense3 = nn.Linear(1024, 8 * 8 * base_channels)

        tensorflow_schedule = [128, 64, 32, 16, 8, 4, 2]
        block_count = out_size.bit_length() - 4  # log2(out_size / 8)
        schedule = tensorflow_schedule[:block_count]
        blocks = []
        input_channels = base_channels
        for output_channels in schedule:
            blocks.extend(
                [
                    nn.ConvTranspose2d(input_channels, output_channels, 4, stride=2, padding=1),
                    nn.SiLU(),
                ]
            )
            input_channels = output_channels
        self.upscale_blocks = nn.Sequential(*blocks)
        self.final_conv = nn.Conv2d(input_channels, 1, 3, padding=1)
        self._initialize_tensorflow_style()

    def _initialize_tensorflow_style(self):
        """Match Keras Dense/Conv Glorot weights and zero-bias defaults."""
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        features = torch.nn.functional.silu(self.dense1(inputs))
        features = torch.nn.functional.silu(self.dense2(features))
        features = self.dense3(features)
        features = self._reshape_latent(features)
        return torch.sigmoid(self.final_conv(self.upscale_blocks(features)))

    def _reshape_latent(self, features):
        # Keras reshapes the flat vector as NHWC (8, 8, C). Convert that exact
        # interpretation to PyTorch NCHW instead of directly viewing it as NCHW.
        return (
            features.view(features.shape[0], 8, 8, self.base_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class HybridLossTorch(nn.Module):
    """PyTorch equivalent of the TensorFlow main model's hybrid loss."""

    def __init__(self, damage_weight=25.0, tversky_weight=1.5, alpha=0.3, beta=0.7):
        super().__init__()
        self.damage_weight = damage_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, solid_probability, solid_truth):
        solid_truth = solid_truth.to(dtype=solid_probability.dtype)
        damage_truth = 1.0 - solid_truth
        epsilon = torch.finfo(solid_probability.dtype).eps
        solid_probability = solid_probability.clamp(epsilon, 1.0 - epsilon)
        damage_probability = 1.0 - solid_probability

        weighted_bce = -(
            solid_truth * torch.log(solid_probability)
            + self.damage_weight * damage_truth * torch.log(damage_probability)
        )
        weighted_bce = weighted_bce.mean()

        axes = tuple(range(1, damage_truth.ndim))
        true_positive = (damage_truth * damage_probability).sum(dim=axes)
        false_positive = ((1.0 - damage_truth) * damage_probability).sum(dim=axes)
        false_negative = (damage_truth * (1.0 - damage_probability)).sum(dim=axes)
        tversky = (true_positive + epsilon) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + epsilon
        )
        return weighted_bce + self.tversky_weight * (1.0 - tversky.mean())


class DamageMetricAccumulator:
    """Global damage-class IoU, Dice, precision, and recall accumulator."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.true_positive = None
        self.false_positive = None
        self.false_negative = None

    @torch.no_grad()
    def update(self, solid_probability, solid_truth):
        damage_prediction = solid_probability < self.threshold
        damage_truth = solid_truth < 0.5
        counts = (
            (damage_prediction & damage_truth).sum(),
            (damage_prediction & ~damage_truth).sum(),
            (~damage_prediction & damage_truth).sum(),
        )
        if self.true_positive is None:
            self.true_positive, self.false_positive, self.false_negative = counts
        else:
            self.true_positive += counts[0]
            self.false_positive += counts[1]
            self.false_negative += counts[2]

    def compute(self):
        if self.true_positive is None:
            tp = fp = fn = 0
        else:
            tp, fp, fn = torch.stack(
                [self.true_positive, self.false_positive, self.false_negative]
            ).cpu().tolist()
        return {
            "damage_iou": tp / max(tp + fp + fn, 1),
            "damage_dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "damage_precision": tp / max(tp + fp, 1),
            "damage_recall": tp / max(tp + fn, 1),
        }
