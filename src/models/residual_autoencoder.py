import torch
from torch import nn
from ..utils.transforms import timestep_embedding
from ..utils.modules import TimestepEmbedSequential, ResidualBlock, Downsample, Upsample


class ResidualAutoEncoder(nn.Module):
    """
    A residual autoencoder that maintains autoencoder structure (encoder-decoder) 
    but uses residual connections within blocks for better gradient flow.
    
    Key differences from UNet:
    1. NO skip connections between encoder and decoder (pure autoencoder)
    2. Uses ResidualBlock for better gradient flow within each layer
    3. Information flows: Input -> Encoder -> Bottleneck -> Decoder -> Output
    4. No concatenation of encoder features with decoder features
    
    This maintains the autoencoder's information bottleneck property while
    improving trainability through residual connections within blocks.
    
    Args:
        in_channels (int): Number of input channels
        model_channels (int): Number of channels in the model's internal representations  
        out_channels (int): Number of output channels
        num_res_blocks (int): Number of residual blocks per stage
        dropout (float): Dropout rate
        channel_mult (tuple): Channel multiplier for each stage
        conv_resample (bool): Whether to use conv for resampling
        norm_groups (int): Number of groups for GroupNorm
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
        self.attention_resolutions = attention_resolutions  # Not used
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads  # Not used
        
        # Time embedding - same as UNet
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        # ENCODER: Build down sampling blocks with residual connections
        # But NO feature saving for skip connections (key autoencoder property)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        )
        current_channels = model_channels
        
        for stage, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # Use ResidualBlock for better gradient flow within each block
                layers = [
                    ResidualBlock(current_channels, mult * model_channels, model_channels * 4, dropout)
                ]
                current_channels = mult * model_channels
                self.encoder_blocks.append(TimestepEmbedSequential(*layers))
            if stage != len(channel_mult) - 1:
                self.encoder_blocks.append(TimestepEmbedSequential(Downsample(current_channels, conv_resample)))
        
        # BOTTLENECK: Middle block with residual connections
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(current_channels, current_channels, model_channels * 4, dropout),
            ResidualBlock(current_channels, current_channels, model_channels * 4, dropout)
        )
        
        # DECODER: Build up sampling blocks with residual connections
        # Key difference: NO concatenation with encoder features (pure autoencoder)
        self.decoder_blocks = nn.ModuleList()
        for stage, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # Use ResidualBlock but with current_channels only (no concat)
                layers = [
                    ResidualBlock(
                        current_channels,  # NO concatenation with encoder features
                        model_channels * mult,
                        model_channels * 4,
                        dropout
                    )
                ]
                current_channels = mult * model_channels
                if stage and i == num_res_blocks:
                    layers.append(Upsample(current_channels, conv_resample))
                self.decoder_blocks.append(TimestepEmbedSequential(*layers))
        
        # Output block
        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups, current_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        Forward pass maintaining pure autoencoder structure:
        Input -> Encoder (with internal residuals) -> Bottleneck -> Decoder (with internal residuals) -> Output
        
        NO skip connections between encoder and decoder layers.
        """
        embedding = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # ENCODER: Process through encoder blocks
        # Note: We do NOT save feature maps for skip connections (autoencoder property)
        current_feature_map = x
        for module in self.encoder_blocks:
            current_feature_map = module(current_feature_map, embedding)
        
        # BOTTLENECK: Process through middle block
        current_feature_map = self.middle_block(current_feature_map, embedding)
        
        # DECODER: Process through decoder blocks
        # Key: NO concatenation with encoder features (maintains autoencoder bottleneck)
        for module in self.decoder_blocks:
            # Pure autoencoder: only pass current feature map, no encoder features
            current_feature_map = module(current_feature_map, embedding)
            
        return self.out(current_feature_map)