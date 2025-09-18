import torch
from torch import nn
from ..utils.transforms import timestep_embedding
from ..utils.modules import TimestepEmbedSequential, TimestepBlock, Downsample, Upsample


class ConvBlock(TimestepBlock):
    """
    Convolutional block with time embedding but without residual connections.
    Similar to ResidualBlock but removes the skip connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.SiLU(),
        norm_groups: int = 32,
    ):
        super().__init__()
        self.time_emb = nn.Sequential(
            activation,
            nn.Linear(time_channels, out_channels)
        )

        self.conv1 = nn.Sequential(
            nn.GroupNorm(norm_groups, in_channels),
            activation,
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_groups, out_channels),
            activation,
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # Channel matching without skip connection - just for dimension adjustment if needed
        self.channel_adjust = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x, t):
        """
        Forward pass for the ConvBlock without residual connection.
        """
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        # No residual connection - just return the processed features
        return h


class AutoEncoderBasic(nn.Module):
    """
    An autoencoder model based on UNetBasic structure but without residual connections.
    
    Uses the same layer structure and channel dimensions as UNetBasic but replaces
    ResidualBlock with ConvBlock (no skip connections) and removes U-Net skip connections.
    
    Args:
        in_channels (int): Number of input channels (e.g., for RGB images, it is 3).
        model_channels (int): Number of channels in the model's internal representations.
        out_channels (int): Number of output channels (e.g., for segmentation, this could be the number of classes).
        num_res_blocks (int): Number of convolutional blocks per downsample and upsample stage.
        dropout (float): Dropout rate to prevent overfitting.
        channel_mult (tuple): Multiplier for the number of channels at each stage.
        conv_resample (bool): Whether to use convolutional layers for resampling.
        norm_groups (int): Number of groups for GroupNorm.
    """
    def __init__(
            self,
            in_channels: int = 2,
            model_channels: int = 128,
            out_channels: int = 1,
            num_res_blocks: int = 2,
            attention_resolutions: tuple = (8, 16),  # Kept for compatibility but not used
            dropout: float = 0.0,
            channel_mult: tuple = (1, 2, 2, 2),
            conv_resample=True,
            num_heads: int = 4,  # Kept for compatibility but not used
            norm_groups: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions  # Not used in basic version
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads  # Not used in basic version
        
        # Time embedding - same as original
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        # Build encoder (down sampling blocks) - without attention and skip connections
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        )
        current_channels = model_channels
        
        for stage, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ConvBlock(current_channels, mult * model_channels, model_channels * 4, dropout)
                ]
                current_channels = mult * model_channels
                self.down_blocks.append(TimestepEmbedSequential(*layers))
            if stage != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(current_channels, conv_resample)))
        
        # Build bottleneck block - without attention and residual connections
        self.middle_block = TimestepEmbedSequential(
            ConvBlock(current_channels, current_channels, model_channels * 4, dropout),
            ConvBlock(current_channels, current_channels, model_channels * 4, dropout)
        )
        
        # Build decoder (up sampling blocks) - without attention and skip connections
        self.up_blocks = nn.ModuleList()
        for stage, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ConvBlock(
                        current_channels,  # No concatenation with skip connections
                        model_channels * mult,
                        model_channels * 4,
                        dropout
                    )
                ]
                current_channels = mult * model_channels
                if stage and i == num_res_blocks:
                    layers.append(Upsample(current_channels, conv_resample))
                self.up_blocks.append(TimestepEmbedSequential(*layers))
        
        # Build output block
        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups, current_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Forward pass of the autoencoder model. Takes an input image and timestep,
        processes it through encoder-decoder architecture without skip connections.

        Args:
            x (torch.Tensor): Input image or feature map.
            timesteps (torch.Tensor): Time step values for dynamic modeling (e.g., for diffusion models).

        Returns:
            torch.Tensor: Processed output image or prediction.
        """
        embedding = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Encoder (down sampling blocks) - no feature map saving for skip connections
        current_feature_map = x
        for module in self.down_blocks:
            current_feature_map = module(current_feature_map, embedding)
        
        # Bottleneck block
        current_feature_map = self.middle_block(current_feature_map, embedding)
        
        # Decoder (up sampling blocks) - no concatenation with skip connections
        for module in self.up_blocks:
            current_feature_map = module(current_feature_map, embedding)
            
        return self.out(current_feature_map)