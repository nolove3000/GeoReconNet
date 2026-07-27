import torch
from torch import nn


ABLATION_VERSION = "formal_architecture_ablation_1024_v1"
VARIANTS = ("single_projection", "light_decoder")


class ArchitectureAblationGenerator(nn.Module):
    """Change one architectural factor relative to the formal high-resolution model."""

    def __init__(self, variant, input_dimension=132, out_size=1024):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if out_size != 1024:
            raise ValueError("Formal architecture ablations require out_size=1024")
        self.variant = variant
        self.input_dimension = input_dimension
        self.out_size = out_size
        self.base_channels = 256 if variant == "single_projection" else 128
        if variant == "single_projection":
            self.embedding = nn.Sequential(nn.Linear(input_dimension, 8 * 8 * self.base_channels))
        else:
            self.embedding = nn.Sequential(
                nn.Linear(input_dimension, 512),
                nn.SiLU(),
                nn.Linear(512, 1024),
                nn.SiLU(),
                nn.Linear(1024, 8 * 8 * self.base_channels),
            )
        blocks = []
        input_channels = self.base_channels
        for output_channels in [128, 64, 32, 16, 8, 4, 2]:
            blocks.extend(
                [nn.ConvTranspose2d(input_channels, output_channels, 4, stride=2, padding=1), nn.SiLU()]
            )
            input_channels = output_channels
        self.decoder = nn.Sequential(*blocks)
        self.output_layer = nn.Conv2d(input_channels, 1, 3, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        latent = self.embedding(inputs)
        latent = latent.view(len(inputs), 8, 8, self.base_channels).permute(0, 3, 1, 2).contiguous()
        return torch.sigmoid(self.output_layer(self.decoder(latent)))
