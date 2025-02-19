
import math
from abc import abstractmethod, ABC
import torch
from torch import nn
from torch.nn import functional as F


class TimestepBlock(nn.Module, ABC):
    """
    Abstract base class for modules that require a timestep embedding as an extra input.

    This class should be subclassed by any module that performs operations conditioned
    on a time embedding.

    Methods:
        forward(x, emb): Processes the input tensor 'x' along with the timestep embedding 'emb'.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the module.

        Args:
            x (torch.Tensor): Input tensor.
            emb (torch.Tensor): Timestep embedding tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential container that passes timestep embeddings to its submodules.

    This container works like nn.Sequential, but if a module is an instance of TimestepBlock,
    it also passes the timestep embedding 'emb' to that module's forward method.

    Methods:
        forward(x, emb): Processes the input tensor 'x' sequentially through all modules,
                           passing 'emb' where required.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies each module in the sequence to the input tensor 'x'.

        Args:
            x (torch.Tensor): Input tensor.
            emb (torch.Tensor): Timestep embedding tensor.

        Returns:
            torch.Tensor: Output tensor after applying all modules.
        """
        for module in self:
            x = module(x, emb) if isinstance(module, TimestepBlock) else module(x)
        return x


# Residual block
class ResidualBlock(TimestepBlock):
    """
    Residual block with a time embedding.

    This block applies a two-stage convolution with group normalization and activation.
    A timestep embedding is incorporated between the convolutions, and a residual shortcut
    connection is added to facilitate gradient flow.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        time_channels (int): Number of channels in the timestep embedding.
        dropout (float, optional): Dropout rate. Default is 0.0.
        activation (nn.Module, optional): Activation function. Default is nn.SiLU().
        norm_groups (int, optional): Number of groups for group normalization. Default is 32.

    Methods:
        forward(x, t): Applies the residual block operations on the input tensor 'x'
                       using the timestep embedding 't'.
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

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        Forward pass for the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Timestep embedding tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# Attention block with shortcut
class AttentionBlock(nn.Module):
    """
    Self-attention block with a residual connection.
    This block computes self-attention over the spatial dimensions of the input tensor.
    It normalizes the input using group normalization, computes queries, keys, and values,
    and applies a scaled dot-product attention mechanism. The result is merged and combined
    with the original input via a residual shortcut.

    Args:
        channels (int): Number of channels in the input tensor.
        num_heads (int, optional): Number of attention heads. Default is 1.
        norm_groups (int, optional): Number of groups for group normalization. Default is 32.

    Methods:
        forward(x): Applies self-attention to the input tensor.
    """
    def __init__(self, channels: int, num_heads: int = 1, norm_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        """
        Forward pass for the AttentionBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with self-attention applied.
        """
        self.B, self.C, self.H, self.W = x.shape
        self.head_dim = self.C // self.num_heads

        qkv = self.qkv(self.norm(x))
        q, k, v = self._extract_qkv(qkv)
        scale = 1. / math.sqrt(math.sqrt(self.head_dim))
        attn = F.softmax(torch.einsum("bct,bcs->bts", q * scale, k * scale), dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = self._merge_heads(h)
        return self.proj(h) + x

    def _extract_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and split the combined query, key, and value tensor into separate components.

        Args:
            x (torch.Tensor): Tensor of shape (B, 3 * C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tensors for query, key, and value.
        """
        return x.reshape(self.B * self.num_heads, -1, self.H * self.W).chunk(3, dim=1)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge the attention heads back into a single tensor.

        Args:
            x (torch.Tensor): Tensor of shape (B * num_heads, C, H * W).

        Returns:
            torch.Tensor: Tensor of shape (B, C, H, W) after merging heads.
        """
        return x.reshape(self.B, -1, self.H, self.W)


class Downsample(nn.Module):
    """
    Downsampling block that reduces the spatial resolution of the input tensor.
    This block applies either a strided convolution or an average pooling operation to
    reduce the height and width of the input tensor.

    Args:
        channels (int): Number of channels in the input tensor.
        use_conv (bool, optional): If True, uses a convolution for downsampling.
                                   Otherwise, uses average pooling. Default is True.

    Methods:
        forward(x): Applies the downsampling operation to the input tensor.
    """
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1) if use_conv else nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for downsampling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Downsampled tensor.
        """
        return self.op(x)


class Upsample(nn.Module):
    """
    Upsampling block that increases the spatial resolution of the input tensor.

    This block first upsamples the input tensor using nearest-neighbor interpolation,
    and then, if desired, applies a convolution to refine the upsampled features.

    Args:
        channels (int): Number of channels in the input tensor.
        use_conv (bool, optional): If True, applies a convolution after upsampling.
                                   Default is True.

    Methods:
        forward(x): Upsamples the input tensor.
    """
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1) if use_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for upsampling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Upsampled tensor, optionally refined by a convolution.
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.conv else x
