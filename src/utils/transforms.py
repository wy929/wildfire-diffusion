from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import math


class SetBorderToZero(object):
    def __init__(self, border_width=1):
        self.border_width = border_width

    def __call__(self, tensor):
        tensor = tensor.clone()
        b, c, h, w = tensor.shape

        # Set top border to zero
        tensor[:, :, :self.border_width, :] = 0
        
        # Set bottom border to zero
        tensor[:, :, -self.border_width:, :] = 0
        
        # Set left border to zero
        tensor[:, :, :, :self.border_width] = 0
        
        # Set right border to zero
        tensor[:, :, :, -self.border_width:] = 0
        
        return tensor


# Define a custom transform class
class ThresholdToZero(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        tensor = tensor.clone()
        tensor[tensor < self.threshold] = 0
        return tensor


class ThresholdToOne(object):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def __call__(self, tensor):
        tensor = tensor.clone()
        tensor[tensor > self.threshold] = 1
        return tensor


class Threshold(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        return (tensor > self.threshold).float()


class Scale0_1To255(object):
    def __call__(self, tensor):
        return (tensor * 255.0).byte()


class Scale255To0_1(object):
    def __call__(self, tensor):
        return tensor / 255.0


class CustomToTensor(object):
    def __call__(self, input):
        if isinstance(input, Image.Image):  # Check if input is a PIL Image
            return self.pil_to_tensor(input)
        elif isinstance(input, np.ndarray):  # Check if input is a NumPy array
            return self.numpy_to_tensor(input)
        elif isinstance(input, torch.Tensor):  # Check if input is a PyTorch tensor
            return input.clone()
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

    def pil_to_tensor(self, pic):
        # Convert PIL Image to Tensor without scaling
        return torch.from_numpy(np.array(pic)).float()

    def numpy_to_tensor(self, array):
        # Convert NumPy array to Tensor without scaling
        return torch.from_numpy(array).float()

transform = transforms.Compose([
    transforms.ToTensor(),
    Threshold(threshold=0.5)
])

post_process = transforms.Compose([
    Threshold(threshold=0.5)
])

def post_transform(threshold=0.5):
    return transforms.Compose([
        CustomToTensor(),
        Scale255To0_1(),
        SetBorderToZero(border_width=3),
        ThresholdToZero(threshold=threshold),
    ])

def post_transform_1(th1=0.5, th2=0.9):
    return transforms.Compose([
        CustomToTensor(),
        Scale255To0_1(),
        SetBorderToZero(border_width=3),
        ThresholdToZero(threshold=th1),
        ThresholdToOne(threshold=th2),
    ])

# define Sinusoidal positional encoding
def timestep_embedding(timesteps, embedding_dim: int, max_frequency: int = 10000):
    """
    Compute sinusoidal positional encoding.
    Args:
        timesteps (torch.Tensor): Timesteps of shape (T,).
        embedding_dim (int): Dimension of the positional encoding.
        max_frequency (float): Maximum frequency for the positional encoding.
    Returns:
        torch.Tensor: Sinusoidal positional encoding of shape (T, dim).
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_frequency) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
