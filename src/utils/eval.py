import torch
from .metrics import *


criterion_mapping = {
    'MSE': MSELoss,
    'L1': L1Loss,
    'PSNR': PSNR,
    'SSIM': SSIM,
    'KL': KLDivergence,
    'Acc': Accuracy,
    'HR': HitRate,
    'CR': CoverageRate,
    'FID': FID,
}

def normal_eval(diffusion, model, val_loader, device):
    """
    Evaluate the model using a standard diffusion loss over the validation dataset.

    This function sets the model to evaluation mode, computes the loss over each 
    batch using the diffusion process, and returns the average loss across the 
    validation set.

    Args:
        diffusion: An instance of the diffusion process that provides methods for 
                   sampling timesteps and computing the loss.
        model (torch.nn.Module): The model being evaluated.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device on which computations are performed.

    Returns:
        float: The average loss over the validation dataset.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)  # x_n
            y = y.to(device)  # x_n+1

            t = diffusion.sample_timesteps(x.shape[0]).to(device)
            loss = diffusion.loss(x, t, y)

            val_loss += loss.item()  # Accumulate the loss

        val_loss = val_loss / len(val_loader)  # Calculate the average loss
        
    model.train()
    return val_loss

def ensemble_eval(criterion_configs, diffusion, model, n_samples, val_loader, post_transforms, device, ensemble_sizes=None):
    """
    Evaluate ensemble outputs from the model using specified criteria.

    For each batch in the validation set, this function generates ensemble outputs 
    using the diffusion process and computes metrics for each post-processing transform. 
    If ensemble_sizes is provided, it also computes metrics for each ensemble result in the list.

    Args:
        criterion_configs (dict): A dictionary mapping metric names to their configuration parameters.
                                  Each value can be a dictionary of parameters or None.
        diffusion: The diffusion process instance that supports ensemble sampling.
        model (torch.nn.Module): The model being evaluated.
        n_samples (int): Number of samples to generate for ensemble evaluation.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        post_transforms (list of callable): List of post-processing functions to apply to ensemble outputs.
        device (torch.device): The device on which computations are performed.
        ensemble_sizes (list, optional): List of ensemble sizes for evaluation. Defaults to None.

    Returns:
        list: A list of dictionaries containing averaged metric values for each post-transform.
              If ensemble_sizes is provided, the metric keys include the ensemble size as a suffix.
    """
    model.eval()
    
    # Initialize criterions with given parameters
    criterions = [criterion_mapping[name](**params) if params else criterion_mapping[name]() 
                  for name, params in criterion_configs.items()]
    
    val_metrics = [{name: 0.0 for name in criterion_configs.keys()} for _ in post_transforms]
    if ensemble_sizes:
        val_metrics_list = [{f"{name}_{size}": 0.0 for name in criterion_configs.keys() for size in ensemble_sizes} for _ in post_transforms]

    with torch.no_grad():
        for i, (x, y, y_) in enumerate(val_loader):
            x = x.to(device)  # x_n
            y = y.to(device)  # x_n+1

            ensemble_output = diffusion.ensemble_sample(x_n=x, n_samples=n_samples, model=model, ensemble_sizes=ensemble_sizes)
            
            if len(ensemble_output) == 3:
                sampled_images, ensemble_result, ensemble_results_list = ensemble_output
                ensemble_results_list_flag = True
            else:
                sampled_images, ensemble_result = ensemble_output
                ensemble_results_list_flag = False

            for idx, post_transform in enumerate(post_transforms):
                transformed_ensemble_result = post_transform(ensemble_result).to(device)

                for criterion, (name, params) in zip(criterions, criterion_configs.items()):
                    # Calculate metrics for the main ensemble result
                    metric_value = criterion(transformed_ensemble_result, y)
                    val_metrics[idx][name] += metric_value

                    # Free memory by moving back to CPU
                    transformed_ensemble_result.cpu()
                
                    # Calculate metrics for the ensemble results list if available
                    if ensemble_results_list_flag:
                        for size, ensemble_res in zip(ensemble_sizes, ensemble_results_list):
                            transformed_ensemble_res = post_transform(ensemble_res).to(device)
                            metric_value_list = criterion(transformed_ensemble_res, y)
                            val_metrics_list[idx][f"{name}_{size}"] += metric_value_list
                            # Free memory by moving back to CPU
                            transformed_ensemble_res.cpu()

        for idx in range(len(post_transforms)):
            val_metrics[idx] = {key: val / len(val_loader) for key, val in val_metrics[idx].items()}
        
            if ensemble_results_list_flag:
                val_metrics_list[idx] = {key: val / len(val_loader) for key, val in val_metrics_list[idx].items()}
                val_metrics[idx].update(val_metrics_list[idx])

    model.train()
    
    return val_metrics

def eval(model, criterion_configs, val_loader, device, post_transform=None):
    """
    Evaluate the model on the validation dataset using specified criteria.

    This function computes various metrics over the validation dataset. An optional post_transform
    can be applied to the model outputs before computing the metrics.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        criterion_configs (dict): A dictionary mapping metric names to their configuration parameters.
                                  Each value can be a dictionary of parameters or None.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device on which computations are performed.
        post_transform (callable, optional): A function to post-process the model outputs before evaluation.
                                             Defaults to None.

    Returns:
        dict: A dictionary mapping metric names to their average values computed over the validation dataset.
    """
    model.eval()
    criterions = [criterion_mapping[name](**params) if params else criterion_mapping[name]() 
                  for name, params in criterion_configs.items()]
    
    val_metrics = {name: 0.0 for name in criterion_configs.keys()}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if len(batch) == 2:
                x, y = batch
                y_ = None
            elif len(batch) == 3:
                x, y, y_ = batch
            else:
                raise ValueError("Unexpected number of elements in the batch: {}".format(len(batch)))

            x = x.to(device)
            y = y.to(device)

            output = model(x, torch.tensor([0]*x.size(0), dtype=torch.long).to(device))
            if post_transform:
                output = post_transform(output)

            for criterion, (name, params) in zip(criterions, criterion_configs.items()):
                metric_value = criterion(output, y)
                val_metrics[name] += metric_value

        val_metrics = {key: val / len(val_loader) for key, val in val_metrics.items()}
        
    model.train()
    return val_metrics
