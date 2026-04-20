import torch
import torch.nn as nn
import timm


class Conv2DBlock(nn.Module):
    """Two-layer Conv-Norm-Activation block used in the decoder."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """UNet decoder block: upsample, concatenate skip connection, refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = Conv2DBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Mamba encoder + UNet-style decoder for 2D multi-class segmentation.

    Input shape:  [B, 1, H, W]
    Output shape: [B, num_classes, H, W]
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, pretrained_encoder: bool = True):
        super().__init__()

        # timm returns intermediate feature maps from shallow to deep.
        self.encoder = timm.create_model(
            "mambaout_small.in1k",
            pretrained=pretrained_encoder,
            features_only=True,
            in_chans=in_channels,
        )
        encoder_channels = self.encoder.feature_info.channels()

        self.decoder4 = DecoderBlock(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.decoder3 = DecoderBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.decoder2 = DecoderBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0])

        self.up_final_1 = nn.ConvTranspose2d(encoder_channels[0], 96, kernel_size=2, stride=2)
        self.conv_final_1 = Conv2DBlock(96, 64)
        self.up_final_2 = nn.ConvTranspose2d(64, 24, kernel_size=2, stride=2)
        self.head = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm mambaout returns NHWC feature maps, so convert them into NCHW.
        features = self.encoder(x)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        e1, e2, e3, e4 = features

        d1 = self.decoder4(e4, e3)
        d2 = self.decoder3(d1, e2)
        d3 = self.decoder2(d2, e1)

        x = self.up_final_1(d3)
        x = self.conv_final_1(x)
        x = self.up_final_2(x)

        return self.head(x)
