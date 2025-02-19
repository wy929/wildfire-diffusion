import torch
from torch import nn
from ..utils.transforms import timestep_embedding
from ..utils.modules import TimestepEmbedSequential, ResidualBlock, AttentionBlock, Downsample, Upsample


# Attention-residual-U-Net model
class UNet(nn.Module):
    """
    A UNet model with attention and residual blocks.

    Args:
        in_channels (int): Number of input channels (e.g., for RGB images, it is 3).
        model_channels (int): Number of channels in the model's internal representations.
        out_channels (int): Number of output channels (e.g., for segmentation, this could be the number of classes).
        num_res_blocks (int): Number of residual blocks per downsample and upsample stage.
        attention_resolutions (tuple): Resolutions at which attention layers are applied.
        dropout (float): Dropout rate to prevent overfitting.
        channel_mult (tuple): Multiplier for the number of channels at each stage.
        conv_resample (bool): Whether to use convolutional layers for resampling.
        num_heads (int): Number of attention heads in the attention block.
        norm_groups (int): Number of groups for GroupNorm.
    """
    def __init__(
            self,
            in_channels: int = 2,
            model_channels: int = 128,
            out_channels: int = 1,
            num_res_blocks: int = 2,
            attention_resolutions: tuple = (8, 16),
            dropout: float = 0.0,
            channel_mult: tuple = (1, 2, 2, 2),
            conv_resample=True,
            num_heads: int = 4,
            norm_groups: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        # build down sampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        )
        down_block_chans = [model_channels]
        current_channels = model_channels
        spatial_resolution_scale = 1
        for stage, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(current_channels, mult * model_channels, model_channels * 4, dropout)
                ]
                current_channels = mult * model_channels
                if spatial_resolution_scale in attention_resolutions:
                    layers.append(AttentionBlock(current_channels, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(current_channels)
            if stage != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(current_channels, conv_resample)))
                down_block_chans.append(current_channels)
                spatial_resolution_scale *= 2
        # build bottleneck block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(current_channels, current_channels, model_channels * 4, dropout),
            AttentionBlock(current_channels, num_heads),
            ResidualBlock(current_channels, current_channels, model_channels * 4, dropout)
        )
        # build up sampling blocks
        self.up_blocks = nn.ModuleList()
        for stage, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        current_channels + down_block_chans.pop(),
                        model_channels * mult,
                        model_channels * 4,
                        dropout
                    )
                ]
                current_channels = mult * model_channels
                if spatial_resolution_scale in attention_resolutions:
                    layers.append(AttentionBlock(current_channels, num_heads=num_heads))
                if stage and i == num_res_blocks:
                    layers.append(Upsample(current_channels, conv_resample))
                    spatial_resolution_scale //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))
        # bulild output block
        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups, current_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Forward pass of the UNet model. Takes an input image and timestep,
        processes it through the network, and returns the output image.

        Args:
            x (torch.Tensor): Input image or feature map.
            timesteps (torch.Tensor): Time step values for dynamic modeling (e.g., for diffusion models).

        Returns:
            torch.Tensor: Processed output image or prediction.
        """
        embedding = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        feature_maps = []
        # down sampling blocks
        current_feature_map = x
        for module in self.down_blocks:
            current_feature_map = module(current_feature_map, embedding)
            feature_maps.append(current_feature_map)
        # bottleneck block
        current_feature_map = self.middle_block(current_feature_map, embedding)
        # up sampling blocks
        for module in self.up_blocks:
            concatenated_features = torch.cat([
                current_feature_map,
                feature_maps.pop()], dim=1)
            current_feature_map = module(concatenated_features, embedding)
        return self.out(current_feature_map)
