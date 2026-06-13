"""
Convolutional Autoencoder (CAE) model with U-Net skip connections for the AANC project.
Encoder compresses speech features into a compact latent representation.
Decoder uses skip connections to reconstruct features with high fidelity.
"""
import torch
import torch.nn as nn
from src.config import LATENT_DIM, FEATURE_DIM, MAX_FRAMES


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class EncoderBlock(nn.Module):
    """Encoder convolutional block with BatchNorm, activation, and SE attention."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se = SqueezeExcite(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder convolutional block with skip connection input."""

    def __init__(self, in_ch, skip_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size, stride, padding, output_padding)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se = SqueezeExcite(out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Crop skip to match x dimensions
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            skip = skip[:, :, :x.size(2), :x.size(3)]
        elif dh < 0 or dw < 0:
            x = x[:, :, :skip.size(2), :skip.size(3)]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class Encoder(nn.Module):
    """Encoder network with U-Net style feature extraction."""

    def __init__(self, feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Encoder blocks: progressively downsample
        self.enc1 = EncoderBlock(1, 32)      # -> (32, H/2, W/2)
        self.enc2 = EncoderBlock(32, 64)     # -> (64, H/4, W/4)
        self.enc3 = EncoderBlock(64, 128)    # -> (128, H/8, W/8)
        self.enc4 = EncoderBlock(128, 256)   # -> (256, H/16, W/16)

        # Compute flattened size
        self._compute_flatten_size()

        # Bottleneck FC
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim),
        )

    def _compute_flatten_size(self):
        """Compute the flattened tensor size after conv layers."""
        dummy = torch.zeros(1, 1, MAX_FRAMES, self.feature_dim)
        with torch.no_grad():
            s1 = self.enc1(dummy)
            s2 = self.enc2(s1)
            s3 = self.enc3(s2)
            s4 = self.enc4(s3)
        self.conv_output_shape = s4.shape[1:]  # (C, H, W)
        self.flatten_size = s4.view(1, -1).shape[1]
        # Store skip connection shapes
        self.skip_shapes = [s1.shape, s2.shape, s3.shape]

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        flat = s4.view(s4.size(0), -1)
        z = self.fc(flat)
        return z, [s1, s2, s3, s4]


class Decoder(nn.Module):
    """Decoder network with U-Net skip connections for high-fidelity reconstruction."""

    def __init__(self, feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM,
                 conv_output_shape=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.conv_output_shape = conv_output_shape

        flatten_size = conv_output_shape[0] * conv_output_shape[1] * conv_output_shape[2]

        # FC from latent to conv shape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, flatten_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder blocks with skip connections
        # dec4: takes enc4 output (256ch) upsampled + enc3 skip (128ch) -> 128ch
        self.dec4 = DecoderBlock(256, 128, 128)
        # dec3: takes dec4 output (128ch) upsampled + enc2 skip (64ch) -> 64ch
        self.dec3 = DecoderBlock(128, 64, 64)
        # dec2: takes dec3 output (64ch) upsampled + enc1 skip (32ch) -> 32ch
        self.dec2 = DecoderBlock(64, 32, 32)

        # dec1: Final upsample without skip (no input-resolution skip available)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final 1x1 conv to produce output
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, z, skips):
        """
        Args:
            z: latent vector (batch, latent_dim)
            skips: list of [s1, s2, s3, s4] from encoder
        """
        x = self.fc(z)
        x = x.view(x.size(0), *self.conv_output_shape)

        # s4 is our starting point (already x), skip connections from s3, s2, s1
        s1, s2, s3, _ = skips

        x = self.dec4(x, s3)   # Upsample + skip from enc3
        x = self.dec3(x, s2)   # Upsample + skip from enc2
        x = self.dec2(x, s1)   # Upsample + skip from enc1

        x = self.dec1(x)       # Final upsample to original resolution

        x = self.final(x)

        # Crop/pad to exact target size
        x = x[:, :, :MAX_FRAMES, :self.feature_dim]
        return x


class ConvAutoencoder(nn.Module):
    """
    U-Net Convolutional Autoencoder for speech feature compression.
    Uses skip connections for high-fidelity reconstruction while
    achieving significant compression through the bottleneck.
    """

    def __init__(self, feature_dim=FEATURE_DIM, latent_dim=LATENT_DIM):
        super().__init__()

        self.encoder = Encoder(feature_dim, latent_dim)
        self.decoder = Decoder(feature_dim, latent_dim,
                               conv_output_shape=self.encoder.conv_output_shape)

        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] U-Net ConvAutoencoder initialized")
        print(f"[Model] Total parameters: {total_params:,}")
        print(f"[Model] Trainable parameters: {trainable_params:,}")
        print(f"[Model] Input shape: (batch, 1, {MAX_FRAMES}, {feature_dim})")
        print(f"[Model] Latent dim: {latent_dim}")
        print(f"[Model] Conv output shape after encoder: {self.encoder.conv_output_shape}")
        print(f"[Model] Compression ratio (features): {MAX_FRAMES * feature_dim / latent_dim:.1f}x")

    def forward(self, x):
        z, skips = self.encoder(x)
        x_recon = self.decoder(z, skips)
        return x_recon, z

    def encode(self, x):
        z, _ = self.encoder(x)
        return z

    def decode(self, z):
        """Decode without skip connections (for inference on standalone latent codes)."""
        # Create dummy skips filled with zeros
        dummy_input = torch.zeros(z.size(0), 1, MAX_FRAMES, self.encoder.feature_dim,
                                  device=z.device)
        _, skips = self.encoder(dummy_input)
        return self.decoder(z, skips)

    def get_compression_ratio(self):
        """Calculate the compression ratio."""
        original_size = MAX_FRAMES * FEATURE_DIM
        compressed_size = LATENT_DIM
        return original_size / compressed_size
