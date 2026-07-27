import torch
from torch import nn


SIMPLE_DECONV_VERSION = "single_projection_deconv_1024_v1"


class SimpleDeconvBaseline(nn.Module):
    """Map modal features to a fine mask with one projection and plain deconvolutions."""

    def __init__(self, input_dimension=132, latent_channels=128, out_size=1024):
        super().__init__()
        if out_size != 1024:
            raise ValueError("The formal simple-deconvolution baseline requires out_size=1024")
        self.input_dimension = input_dimension
        self.latent_channels = latent_channels
        self.out_size = out_size
        self.projection = nn.Linear(input_dimension, 8 * 8 * latent_channels)
        channel_schedule = [128, 64, 32, 16, 8, 4, 2]
        blocks = []
        input_channels = latent_channels
        for output_channels in channel_schedule:
            blocks.extend(
                [
                    nn.ConvTranspose2d(
                        input_channels, output_channels, kernel_size=4, stride=2, padding=1
                    ),
                    nn.SiLU(),
                ]
            )
            input_channels = output_channels
        self.decoder = nn.Sequential(*blocks)
        self.output_layer = nn.Conv2d(input_channels, 1, kernel_size=3, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        latent = self.projection(inputs)
        latent = latent.view(len(inputs), 8, 8, self.latent_channels).permute(0, 3, 1, 2)
        return torch.sigmoid(self.output_layer(self.decoder(latent.contiguous())))
