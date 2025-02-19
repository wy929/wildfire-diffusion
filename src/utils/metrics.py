import torch
from torch import nn
from skimage.metrics import structural_similarity as ssim
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms.functional import resize
from torch.nn.functional import adaptive_avg_pool2d


def calculate_avg_loss(predicted, target, loss_type='mse', device='cuda'):
    predicted = predicted.to(device)
    target = target.to(device)

    if loss_type == 'mse':
        loss_fn = nn.MSELoss(reduction='none').to(device)
    elif loss_type == 'l1':
        loss_fn = nn.L1Loss(reduction='none').to(device)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    losses = loss_fn(predicted, target).view(predicted.shape[0], -1).mean(dim=1)
    avg_loss = losses.mean().item()

    return avg_loss


def calculate_ssim(predicted, target, win_size=7):
    # Ensure the inputs are torch tensors and move to CPU if necessary
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Initialize an empty list to store SSIM values
    ssim_values = []

    # Check the shape of the inputs and handle accordingly
    if predicted.ndim == 4:  # Shape [batch, channels, 64, 64]
        for i in range(predicted.shape[0]):
            for j in range(predicted.shape[1]):
                ssim_val = ssim(predicted[i, j], target[i, j], win_size=win_size, data_range=1.0)
                ssim_values.append(ssim_val)
    elif predicted.ndim == 3:  # Shape [channels, 64, 64]
        for j in range(predicted.shape[0]):
            ssim_val = ssim(predicted[j], target[j], win_size=win_size, data_range=1.0)
            ssim_values.append(ssim_val)
    elif predicted.ndim == 2:  # Shape [64, 64]
        ssim_val = ssim(predicted, target, win_size=win_size, data_range=1.0)
        ssim_values.append(ssim_val)
    else:
        raise ValueError("Unsupported input shape. Expected shapes are [batch, channels, 64, 64], [channels, 64, 64], or [64, 64].")

    return np.mean(ssim_values)


def calculate_psnr(predicted, target, device='cuda'):
    predicted = predicted.to(device)
    target = target.to(device)
    mse = nn.MSELoss()(predicted, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def calculate_mse(predicted, target, device='cuda'):
    predicted = predicted.to(device)
    target = target.to(device)
    mse = nn.MSELoss()(predicted, target)
    return mse.item()


def calculate_accuracy(predicted, target, threshold=0.5, device='cuda'):
    predicted = (predicted.to(device) > threshold).float()
    target = (target.to(device) > threshold).float()
    correct = (predicted == target).float().sum()
    total = torch.numel(predicted)
    accuracy = correct / total
    return accuracy.item()


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predicted, target, device='cuda'):
        predicted = predicted.to(device)
        target = target.to(device)
        mse = self.loss_fn(predicted, target)
        return mse.item()


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, predicted, target, device='cuda'):
        predicted = predicted.to(device)
        target = target.to(device)
        l1 = self.loss_fn(predicted, target)
        return l1.item()


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, predicted, target, device='cuda'):
        predicted = predicted.to(device)
        target = target.to(device)
        mse = nn.MSELoss()(predicted, target)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()


class SSIM(nn.Module):
    def __init__(self, win_size=7):
        super(SSIM, self).__init__()
        self.win_size = win_size

    def forward(self, predicted, target):
        # Ensure the inputs are torch tensors and move to CPU if necessary
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        # Initialize an empty list to store SSIM values
        ssim_values = []

        # Check the shape of the inputs and handle accordingly
        if predicted.ndim == 4:  # Shape [batch, channels, 64, 64]
            for i in range(predicted.shape[0]):
                for j in range(predicted.shape[1]):
                    ssim_val = ssim(predicted[i, j], target[i, j], win_size=self.win_size, data_range=1.0)
                    ssim_values.append(ssim_val)
        elif predicted.ndim == 3:  # Shape [channels, 64, 64]
            for j in range(predicted.shape[0]):
                ssim_val = ssim(predicted[j], target[j], win_size=self.win_size, data_range=1.0)
                ssim_values.append(ssim_val)
        elif predicted.ndim == 2:  # Shape [64, 64]
            ssim_val = ssim(predicted, target, win_size=self.win_size, data_range=1.0)
            ssim_values.append(ssim_val)
        else:
            raise ValueError("Unsupported input shape. Expected shapes are [batch, channels, 64, 64], [channels, 64, 64], or [64, 64].")

        return np.mean(ssim_values)


class KLDivergence(nn.Module):
    def __init__(self, reduction='batchmean', epsilon=1e-10):
        super(KLDivergence, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.loss_fn = nn.KLDivLoss(reduction=self.reduction)

    def forward(self, predicted, target, device='cuda'):
        predicted = predicted.to(device)
        target = target.to(device)

        # Ensure the predicted values are log-probabilities
        predicted_log = torch.log(predicted + self.epsilon)

        # Ensure the target values are probabilities
        target_prob = target + self.epsilon

        kl_div = self.loss_fn(predicted_log, target_prob)

        return kl_div.item()


class Accuracy(nn.Module):
    def __init__(self, threshold=0.5):
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def forward(self, predicted, target, device='cuda'):
        predicted = (predicted.to(device) > self.threshold).float()
        target = (target.to(device) > self.threshold).float()
        correct = (predicted == target).float().sum()
        total = torch.numel(predicted)
        accuracy = correct / total
        return accuracy.item()


class HitRate(nn.Module):
    def __init__(self, threshold=0.2):
        super(HitRate, self).__init__()
        self.threshold = threshold

    def forward(self, predicted, target, device='cuda'):
        predicted = predicted.to(device)
        target = target.to(device)

        # Mask to only consider values where target > 0
        mask = target > 0

        # Calculate the absolute difference
        difference = torch.abs(predicted - target)

        # Calculate hits where the difference is less than the threshold
        hits = (difference < self.threshold) & mask

        # Count the number of hits and the number of valid target elements
        num_hits = hits.float().sum()
        num_valid_targets = mask.float().sum()

        # Calculate the hit rate
        hit_rate = num_hits / num_valid_targets if num_valid_targets > 0 else torch.tensor(0.0, device=device)

        return hit_rate.item()


class CoverageRate(nn.Module):
    def __init__(self, pth=0.05, threshold=0.1):
        super(CoverageRate, self).__init__()
        self.threshold = threshold
        self.pth = pth

    def forward(self, predicted, target, device='cuda'):
        predicted = predicted.to(device)
        predicted = (predicted > self.pth).float()
        target = target.to(device)
        target = (target > 0).float()

        # Mask to only consider values where target > 0
        mask = target > 0

        # Calculate the absolute difference
        difference = torch.abs(predicted - target)

        # Calculate covered where the difference is less than the threshold
        covered = (difference < self.threshold) & mask

        # Count the number of covered and the number of valid target elements
        num_covered = covered.float().sum()
        num_valid_targets = mask.float().sum()

        # Calculate the coverage rate
        coverage_rate = num_covered / num_valid_targets if num_valid_targets > 0 else torch.tensor(0.0, device=device)

        return coverage_rate.item()


class FID(nn.Module):
    def __init__(self, device='cuda', resize_to=(299, 299)):
        super(FID, self).__init__()
        self.device = device
        self.resize_to = resize_to
        inception = inception_v3(transform_input=False).to(device)
        inception.load_state_dict(torch.load('./Downloads/models/inception_v3_weights.pth'))
        inception.to(device)
        self.inception_model = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            inception.maxpool1,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            inception.maxpool2,
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        ).to(device)
        self.inception_model.eval()

    def preprocess_images(self, images):
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        images = resize(images, self.resize_to)
        return images

    def get_activations(self, images):
        with torch.no_grad():
            images = self.preprocess_images(images)
            images = images.to(self.device)
            features = self.inception_model(images)
            features = adaptive_avg_pool2d(features, (1, 1))
            return features.squeeze(-1).squeeze(-1)

    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2

        sigma1 += torch.eye(sigma1.size(0), device=self.device) * eps
        sigma2 += torch.eye(sigma2.size(0), device=self.device) * eps

        covmean = matrix_sqrt_torch(sigma1 @ sigma2)

        fid = diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)
        return fid.item()

    def forward(self, real_images, generated_images):
        real_activations = self.get_activations(real_images)
        gen_activations = self.get_activations(generated_images)

        mu1 = torch.mean(real_activations, dim=0)
        sigma1 = torch.cov(real_activations.T)

        mu2 = torch.mean(gen_activations, dim=0)
        sigma2 = torch.cov(gen_activations.T)

        fid = self.calculate_fid(mu1, sigma1, mu2, sigma2)
        return fid


# helper function to calculate matrix square root
def matrix_sqrt_torch(matrix, eps=1e-6):
    error = False
    matrix = (matrix + matrix.T) / 2
    try:
        # using eigh for symmetric matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0) + eps)
        matrix_sqrt = (eigenvectors * sqrt_eigenvalues.unsqueeze(0)) @ eigenvectors.T
    except torch._C._LinAlgError:
        try:
            # if eigh did not converge, use SVD
            U, S, V = torch.linalg.svd(matrix)
            S_sqrt = torch.sqrt(S)
            matrix_sqrt = U @ torch.diag(S_sqrt) @ V.T
        except Exception as e:
            # if SVD also did not converge, use the approximation
            matrix_sqrt = torch.diag(torch.sqrt(torch.clamp(torch.diag(matrix), min=0)))
            if error is True:
                print(f"Error in matrix square root calculation) {e}")
    return matrix_sqrt
