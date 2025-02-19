import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class Diffusion(nn.Module):
    """
    Diffusion model class that implements the diffusion and denoising process.

    This class includes methods for:
        - Diffusion process (forward pass with noise addition)
        - Denoising score matching (loss computation)
        - Sampling from the diffusion model (DDPM and DDIM methods)
        - Post-processing of the generated images

    Args:
        model (nn.Module): The neural network model to be used for denoising (e.g., U-Net).
        T (int): The number of diffusion timesteps (default is 500).
        beta_start (float): The starting value of beta in the diffusion schedule (default is 1e-4).
        beta_end (float): The ending value of beta in the diffusion schedule (default is 0.02).
        img_size (int): The size of the images (default is 64).
        loss_type (str): The type of loss used for training (default is 'l2').
        device (str): The device to use for computation (default is 'cpu').
        steps (int): The number of steps for DDIM sampling (default is 500).
        post_process (callable, optional): A function to post-process the generated images (default is None).
    """
    def __init__(
        self,
        model,
        T: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        img_size: int = 64,
        loss_type: str = "l2",
        device: str = "cpu",
        steps: int = 500,
        post_process=None
    ):
        super().__init__()
        self.model = model
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.loss_type = loss_type
        self.device = device
        self.steps = steps
        self.post_process = post_process
        self.beta = self.noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alphas_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alpha_bar)
        self.coeff_1 = 1.0 / torch.sqrt(self.alpha)
        self.coeff_2 = self.coeff_1 * ((1.0 - self.alpha) / torch.sqrt(1.0 - self.alpha_bar))
        self.sigma_t = torch.sqrt(self.beta)

    def noise_schedule(self):
        """
        Creates a linear noise schedule based on `beta_start` and `beta_end`.

        Returns:
            torch.Tensor: A tensor of size (T,) representing the noise schedule over `T` timesteps.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.T)

    def perturb_x(self, x_0, t, epsilon):
        """
        Perturbs the input image `x_0` with noise according to the diffusion process at timestep `t`.

        Args:
            x_0 (torch.Tensor): The original image (B x C x H x W).
            t (torch.Tensor): The timestep tensor (B,).
            epsilon (torch.Tensor): The noise tensor to be added.

        Returns:
            torch.Tensor: The perturbed image `x_t`.
            torch.Tensor: The noise `epsilon`.
        """
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * epsilon
        return x_t, epsilon

    def loss(self, x, t, y, criterion=False):
        """
        Computes the loss for the denoising score matching task.

        Args:
            x (torch.Tensor): The noisy image at timestep `t` (x_n).
            t (torch.Tensor): The timestep tensor (B,).
            y (torch.Tensor): The true image at timestep `t+1` (x_{n+1}).
            criterion (callable, optional): The loss function to be used (default is `None`).

        Returns:
            torch.Tensor: The computed loss.
        """
        epsilon = torch.randn_like(y)
        y_t, epsilon = self.perturb_x(y, t, epsilon)
        y_t = torch.cat([y_t, x], dim=1)  # B x (channel*2) x H x W
        epsilon_theta = self.model(y_t, t)

        if not criterion:
            if self.loss_type == "l1":
                loss = F.l1_loss(epsilon_theta, epsilon)
            elif self.loss_type == "l2":
                loss = F.mse_loss(epsilon_theta, epsilon)
        else:
            loss = criterion(epsilon_theta, epsilon)

        return loss

    @torch.no_grad()
    def sample_timesteps(self, n):
        """
        Samples random timesteps for the batch of images.

        Args:
            n (int): The batch size.

        Returns:
            torch.Tensor: A tensor of sampled timesteps (n,).
        """
        return torch.randint(low=1, high=self.T, size=(n,))

    @torch.no_grad()
    def ddpm_sample(self, x_n, model=None, clamp=1, post_process=None):
        """
        Perform DDPM (Denoising Diffusion Probabilistic Models) sampling.

        Args:
            x_n (torch.Tensor): The noisy image to start the reverse diffusion process (B x 1 x H x W).
            model (nn.Module, optional): The denoising model (default is the `self.model`).
            clamp (int, optional): Whether to clamp the output (default is 1).
            post_process (callable, optional): A function to post-process the output (default is `None`).

        Returns:
            torch.Tensor: The generated image after reverse diffusion.
        """
        n = x_n.shape[0]
        if model is None:
            model = self.model
        model.eval()
        with torch.no_grad():
            x_t = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)  # B x 1 x H x W
            with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T, position=0) as sampling_steps:
                for time_step in sampling_steps:
                    t = torch.full((x_n.shape[0],), time_step, device=self.device, dtype=torch.long)  # B
                    z = torch.randn_like(x_t) if time_step > 1 else torch.zeros_like(x_t)  # B x 1 x H x W
                    x_in = torch.cat([x_t, x_n], dim=1)  # B x (channel*2) x H x W
                    epsilon_theta = model(x_in, t)
                    coeff_1 = extract(self.coeff_1, t, x_n.shape)
                    coeff_2 = extract(self.coeff_2, t, x_n.shape)
                    sigma_t = extract(self.sigma_t, t, x_n.shape)
                    x_t = coeff_1 * x_t - coeff_2 * epsilon_theta + sigma_t * z
        model.train()
        if clamp == 0:
            x_t = (x_t.clamp(-1, 1) + 1) / 2
            x_t = (x_t * 255).type(torch.uint8)
        elif clamp == 1:
            x_t = x_t.clamp(0, 1)
            x_t = (x_t * 255).type(torch.uint8)
        return x_t

    @torch.no_grad()
    def ensemble_sample(self, x_n, n_samples=1, model=None, ensemble_sizes=None):
        """
        Perform ensemble sampling where multiple samples are averaged to form the final image.

        Args:
            x_n (torch.Tensor): The noisy input image (B x 1 x H x W).
            n_samples (int): The number of samples to generate (default is 1).
            model (nn.Module, optional): The denoising model (default is `self.model`).
            ensemble_sizes (list, optional): A list of sizes for ensemble averaging (default is `None`).

        Returns:
            tuple: A tuple containing:
                - A list of generated samples
                - The ensemble result (average of all samples)
                - A list of ensemble results for different ensemble sizes (if provided).
        """
        if model is None:
            model = self.model
        x_n = x_n.to(self.device)

        sampled_images = []
        for i in range(n_samples):
            sampled_images.append(self.forward(x_n=x_n, model=model).cpu())

        sampled_images_ = torch.stack(sampled_images)  # n_samples x B x C x H x W
        sampled_images_ = sampled_images_.float().to(self.device)  # n_samples x B x C x H x W
        ensemble_result = sampled_images_.mean(dim=0).cpu()  # B x C x H x W

        ensemble_results_list = []
        if ensemble_sizes:
            for size in ensemble_sizes:
                if size <= n_samples:
                    ensemble_mean = sampled_images_[:size].mean(dim=0).cpu()
                    ensemble_results_list.append(ensemble_mean)

        if ensemble_sizes:
            return sampled_images, ensemble_result, ensemble_results_list
        return sampled_images, ensemble_result

    @torch.no_grad()
    def ddim_sample(self, x_n, model=None, steps=None, eta=0.0, method="linear", only_return_x_0=True, interval=1, clamp=1, post_process=None):
        """
        DDIM sampling method that generates samples from the diffusion model.

        Args:
            x_n (torch.Tensor): The input noisy image or tensor to start the sampling process. It is typically the noisy image at a certain timestep.
            model (nn.Module, optional): The model used for denoising. If not provided, the model from the class initialization is used.
            steps (int, optional): The number of sampling steps to take (default is None, uses self.steps).
            eta (float, optional): The noise weight factor (default is 0.0). Controls the trade-off between deterministic and stochastic denoising.
            method (str, optional): The method for sampling, either 'linear' or 'quadratic' (default is 'linear'). This affects how the timesteps
                are chosen during sampling.
            only_return_x_0 (bool, optional): If True, only the final image (x_0) will be returned. If False, intermediate steps will
                be included (default is True).
            interval (int, optional): The interval for returning intermediate steps (default is 1). Only used if only_return_x_0 is False.
            clamp (int, optional): Determines how to clamp the output image. If 0, the values will be scaled to [-1, 1]. If 1, the values
            will be clamped to [0, 1] (default is 1).
            post_process (callable, optional): A function to apply any post-processing to the generated images (default is None).

        Returns:
            torch.Tensor:
                The generated image (x_0) after sampling. Depending on the settings, it could be clamped to the specified range and/or post-processed.

        Example:
            ```python
            x_0 = model.ddim_sample(noisy_image, eta=0.1, method='linear', steps=500)
            ```

        Raises:
            NotImplementedError:
                If the provided sampling method is not implemented (currently only 'linear' and 'quadratic' are supported).
        """
        if steps is None:
            steps = self.steps
        if model is None:
            model = self.model
        model.eval()
        with torch.no_grad():
            x_t = torch.randn((x_n.shape[0], 1, self.img_size, self.img_size)).to(self.device)  # B x 1 x H x W

            if method == "linear":
                a = self.T // steps
                time_steps = np.asarray(list(range(0, self.T, a)))
            elif method == "quadratic":
                time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int)
            else:
                raise NotImplementedError(f"sampling method {method} is not implemented!")

            time_steps = time_steps + 1
            time_steps_prev = np.concatenate([[0], time_steps[:-1]])

            # x = [x_t]
            with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
                for i in sampling_steps:
                    t = torch.full((x_t.shape[0],), time_steps[i], device=self.device, dtype=torch.long)
                    prev_t = torch.full((x_t.shape[0],), time_steps_prev[i], device=self.device, dtype=torch.long)

                    alpha_t = extract(self.alpha_bar, t, x_t.shape)
                    alpha_t_prev = extract(self.alpha_bar, prev_t, x_t.shape)

                    x_in = torch.cat([x_t, x_n], dim=1)  # B x (channel*2) x H x W
                    epsilon_theta_t = model(x_in, t)

                    sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                    epsilon_t = torch.randn_like(x_t)
                    x_t = (
                        torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                        (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                            (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                        sigma_t * epsilon_t
                    )
                    sampling_steps.set_postfix(ordered_dict={"step": i + 1})
            model.train()
            if post_process is None:
                post_process = self.post_process
                if post_process is not None:
                    x_t = post_process(x_t)
            if clamp == 0:
                x_t = (x_t.clamp(-1, 1) + 1) / 2
                x_t = (x_t * 255).type(torch.uint8)
            elif clamp == 1:
                x_t = x_t.clamp(0, 1)
                x_t = (x_t * 255).type(torch.uint8)
            return x_t

    def forward(self, x_n, model=None, sample='ddim', post_process=None):
        if model is None:
            model = self.model
        if sample == 'ddpm':
            return self.ddpm_sample(x_n, model, post_process=post_process)
        elif sample == 'ddim':
            return self.ddim_sample(x_n, model, post_process=post_process)
        else:
            raise ValueError(f"Unknown sample method: {sample}")


# helper function
def extract(v, i, shape):
    """
    Extract values from a tensor `v` at indices specified by `i`.

    Args:
        v (torch.Tensor): A tensor (typically a noise schedule or coefficients) with shape (T,).
        i (torch.Tensor): A tensor of indices, typically with shape (batch_size,).
        shape (tuple): The desired shape of the output tensor. The function reshapes
                       the output to match this shape after extraction.

    Returns:
        torch.Tensor: A tensor of values extracted from `v` at the indices `i`, reshaped to match the desired `shape`.
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out
