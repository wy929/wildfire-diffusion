from .metrics import *
from .eval import normal_eval, ensemble_eval, eval
from .tools import *


def save_sample_images(diffusion, dataloader, results_dir, epoch, device, num_samples=4):
    imgs, imgs_plus_1 = next(iter(dataloader))
    imgs = imgs[:num_samples].to(device)
    imgs_plus_1 = imgs_plus_1[:num_samples].to(device)
    sampled_images = diffusion(x_n=imgs)
    save_imgs(imgs, results_dir, f"{epoch}_xn", scale=True)
    save_imgs(imgs_plus_1, results_dir, f"{epoch}_xn_plus_1", scale=True)
    save_imgs(sampled_images, results_dir, f"{epoch}")

def save_ensemble_sample_images(diffusion, dataloader, results_dir, epoch, device, num_samples=1, ensemble_samples=5, post_transforms=None, save_original_updates=False):
    # Create a subdirectory for the current epoch
    epoch_dir = os.path.join(results_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Get the first batch from the dataloader
    imgs, ensemble_images, update_images_list = next(iter(dataloader))
    imgs = imgs[:num_samples].to(device)
    ensemble_images = ensemble_images[:num_samples].to(device)
    update_images_list = [update_images[:num_samples].to(device) for update_images in update_images_list]
    
    # Generate ensemble samples
    sampled_images_list, ensemble_result = diffusion.ensemble_sample(x_n=imgs, n_samples=ensemble_samples)
    
    # Save original images
    save_imgs(imgs, epoch_dir, "original", scale=True)
    
    # Save ensemble target images
    save_imgs(ensemble_images, epoch_dir, "ensemble_target", scale=True)
    
    # Save individual update images
    if save_original_updates:
        for i, update_images in enumerate(update_images_list):
            save_imgs(update_images, epoch_dir, f"update_image_{i}", scale=True)
    
    # Save individual sampled images
    for i, sampled_images in enumerate(sampled_images_list):
        save_imgs(sampled_images, epoch_dir, f"sample_{i}")
    
    # Save ensemble averaged images for each post_transform
    if post_transforms:
        for idx, post_transform in enumerate(post_transforms):
            transformed_ensemble_result = post_transform(ensemble_result)
            save_imgs(transformed_ensemble_result, epoch_dir, f"ensemble_post_transform_{idx}", scale=True)
    else:
        save_imgs(ensemble_result, epoch_dir, "ensemble", scale=True)

def train(
        diffusion,
        model, 
        train_loader, 
        optimizer, 
        n_epochs, 
        loss_type='l2',
        criterion=False,
        device='cpu', 
        log_dir='./logs/',
        checkpoint=None,   
        checkpoint_dir='checkpoints',
        checkpoint_interval=10,
        model_name='model',
        results_dir='./results',
        results_interval=10,
        patience=5, 
        normal_val_loader=False,
        normal_val_interval=10,
        ensemble_val_loader=False,
        ensemble_val_interval=10,
        ensemble_val_metrics=[],
        n_samples=4,
        ensemble_sizes=None,
        post_transforms=None,
        config=False):
    best_en_mse = float('inf')
    current_en_mse = float('inf')
    best_epoch = 0
    # set diffusion loss type
    if loss_type:
        diffusion.loss_type = loss_type
    if config:
        os.makedirs(config['root_dir'], exist_ok=True)
        config_path = os.path.join(config['root_dir'], 'config.yml')
        # save config to yaml file
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    # Create the results and checkpoints directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Initialize the logger
    log_file = os.path.join(log_dir, 'train.log')
    logger = setup_logging(log_file)
    # Initialize log dictionaries and CSV files for each post_transform
    log_dicts = []
    csv_files = []
    
    for idx, post_transform in enumerate(post_transforms):
        log_dict = {'epoch': None, 'train_loss': None}
        if normal_val_loader:
            log_dict['n_val_loss'] = None
        if ensemble_val_loader:
            for metric in ensemble_val_metrics:
                log_dict[f'en_{metric}'] = None
            # Initialize additional ensemble sizes
            if ensemble_sizes:
                for size in ensemble_sizes:
                    for metric in ensemble_val_metrics:
                        log_dict[f'en_{metric}_{size}'] = None
        log_dicts.append(log_dict)
        
        csv_file_path = os.path.join(log_dir, f"log_{idx}.csv")
        csv_files.append(csv_file_path)
        initialize_csv(csv_file_path, log_dict.keys())
        
    start_epoch = 1
    end_epoch = n_epochs
    # Load the checkpoint if specified
    if checkpoint:
        start_epoch = load_checkpoint(checkpoint, model, optimizer)
        logger.info(f"Resuming training from epoch {start_epoch}")
        end_epoch = start_epoch + n_epochs
    # Training loop
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        train_loss = 0.0  # Initialize the epoch loss
        val_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)  # x_n
            y = y.to(device)  # x_n+1

            t = diffusion.sample_timesteps(x.shape[0]).to(device)
            if criterion:
                loss = diffusion.loss(x, t, y, criterion)
            else:
                loss = diffusion.loss(x, t, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # Accumulate the loss

        train_loss = train_loss / len(train_loader)  # Calculate the average loss
        
        # Update log dictionaries
        for log_dict in log_dicts:
            log_dict['epoch'] = epoch
            log_dict['train_loss'] = train_loss
            
        # Save sample images
        if epoch > 0 and epoch % results_interval == 0:
            # save_sample_images(diffusion, normal_val_loader, results_dir, epoch, device, num_samples=4)
            if ensemble_val_loader:
                save_ensemble_sample_images(diffusion, ensemble_val_loader, results_dir, epoch, device, num_samples=8, ensemble_samples=n_samples, post_transforms=post_transforms)

        # Normal validation
        if normal_val_loader:
            if epoch > 0 and epoch % normal_val_interval == 0:
                val_loss = normal_eval(diffusion, model, normal_val_loader, device)
                for log_dict in log_dicts:
                    log_dict['n_val_loss'] = val_loss
        
        # Ensemble validation
        if ensemble_val_loader:
            if epoch > 0 and epoch % ensemble_val_interval == 0:
                val_metrics_list = ensemble_eval(
                    ensemble_val_metrics, diffusion, model, n_samples=n_samples, val_loader=ensemble_val_loader, 
                    post_transforms=post_transforms, device=device, ensemble_sizes=ensemble_sizes)
                # print(val_metrics_list)
                
                for idx, val_metrics in enumerate(val_metrics_list):
                    log_dict = log_dicts[idx]
                    for metric in ensemble_val_metrics:
                        log_dict[f'en_{metric}'] = val_metrics[metric]
                        # Log additional ensemble sizes if specified
                        if ensemble_sizes:
                            for size in ensemble_sizes:
                                log_dict[f'en_{metric}_{size}'] = val_metrics[f'{metric}_{size}'] 
                    if (idx==0) and ('MSE' in ensemble_val_metrics):
                        # current_en_mse = log_dicts[0]["en_MSE"]
                        current_en_mse = val_metrics['MSE']

        # Save logs and CSV files for each post_transform
        for idx, (log_dict, csv_file_path) in enumerate(zip(log_dicts, csv_files)):
            save_to_csv(csv_file_path, log_dict)
            log_non_none_items(logger, log_dict, idx)  # Log the post_transform log_dict with idx
        # Checkpoint save
        if epoch > 0 and epoch % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, model_name, csv_files)
        # Save best checkpoint  
        if current_en_mse < best_en_mse:
            best_en_mse = current_en_mse
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, model_name, csv_files, best=True)
            logger.info(f"New best model saved at epoch {epoch} with en_MSE: {best_en_mse}")
        # Reset log dictionaries
        log_dicts = [{k: None for k in log_dict.keys()} for log_dict in log_dicts]

def train_normal(
          model,
          train_loader, 
          optimizer, 
          n_epochs, 
          criterion='l2', 
          device='cpu',  
          log_file='./logs/training.log', 
          csv_file='./logs/training.csv', 
          checkpoint=None,   
          checkpoint_dir='checkpoints', 
          checkpoint_interval=10,
          model_name='model',
          results_dir='./results',
          results_interval=10,
          patience=5,
          normal_val_loader=False,
          normal_val_interval=10,
          normal_val_metrics=[],
          ensemble_val_loader=False,
          ensemble_val_interval=10,
          ensemble_val_metrics=[],
          n_samples=1,
          post_transform=None,
          config=False):
    if config:
        os.makedirs(config['root_dir'], exist_ok=True)
        config_path = os.path.join(config['root_dir'], 'config.yml')
        # save config to yaml file
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    # Create the results and checkpoints directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Initialize the logger
    logger = setup_logging(log_file)
    log_dict = {'epoch': None, 'train_loss': None}
    if normal_val_loader:
        for metric in normal_val_metrics:
            log_dict[f'n_{metric}'] = None
    if ensemble_val_loader:
        for metric in ensemble_val_metrics:
            log_dict[f'en_{metric}'] = None
    # Initialize the CSV file
    initialize_csv(csv_file, log_dict.keys())
    start_epoch = 1
    end_epoch = n_epochs
    # Load the checkpoint if specified
    if checkpoint:
        start_epoch = load_checkpoint(checkpoint, model, optimizer)
        logger.info(f"Resuming training from epoch {start_epoch}")
        end_epoch = start_epoch + n_epochs
    # Function to visualize the results
    def visualize_results(epoch, model, data_loader, device, results_dir, num_samples=8, plot=False):
        model.eval()
        batch = next(iter(data_loader))
        if len(batch) == 2:
            inputs, targets = batch
            # inputs, targets = inputs.to(device), targets.to(device)
        elif len(batch) == 3:
            inputs, targets, update_images_list = batch
            # inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs[:num_samples].to(device)
        targets = targets[:num_samples].to(device)
        outputs = model(inputs, torch.tensor([0]*inputs.size(0), dtype=torch.long).to(device))
        if post_transform:
            outputs = post_transform(outputs)
        if not plot:
            save_imgs(inputs, results_dir, f"{epoch}_xn", scale=True)
            save_imgs(targets, results_dir, f"{epoch}_xn_plus_1", scale=True)
            save_imgs(outputs, results_dir, f"{epoch}", scale=True)
        else:
            # Convert tensors to numpy arrays for visualization
            inputs = inputs.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            # Determine the number of samples to plot
            num_samples = inputs.shape[0]

            # Calculate the number of columns for the plot grid
            num_cols = min(5, num_samples)
            num_rows = 3  # We have three rows: input, target, output

            # Plot the results
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 9))
            
            for i in range(num_samples):
                col = i % num_cols
                if num_samples > num_cols:
                    row = i // num_cols
                else:
                    row = 0
                    
                if num_rows == 1:
                    axs[col].imshow(inputs[i][0], cmap='gray')
                    axs[col].set_title('Input')
                    axs[col].axis('off')
                else:
                    axs[0, col].imshow(inputs[i][0], cmap='gray')
                    axs[0, col].set_title('Input')
                    axs[0, col].axis('off')

                    axs[1, col].imshow(targets[i][0], cmap='gray')
                    axs[1, col].set_title('Target')
                    axs[1, col].axis('off')

                    axs[2, col].imshow(outputs[i][0], cmap='gray')
                    axs[2, col].set_title('Output')
                    axs[2, col].axis('off')
            
            plt.suptitle(f'Epoch {epoch}')
            plt.savefig(os.path.join(results_dir, f'epoch_{epoch}.png'))
            plt.close()
    # Training loop with visualization
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        train_loss = 0.0  # Initialize the epoch loss
        val_loss = None
        ensemble_val_loss = None

        for frame_n, frame_n_plus_1 in train_loader:
            frame_n, frame_n_plus_1 = frame_n.to(device), frame_n_plus_1.to(device)

            # Forward pass
            outputs = model(frame_n, torch.tensor([0]*frame_n.size(0), dtype=torch.long).to(device))
            loss = criterion(outputs, frame_n_plus_1)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # Accumulate the loss
        
        train_loss = train_loss / len(train_loader)  # Calculate the average loss
        log_dict['epoch'] = epoch
        log_dict['train_loss'] = train_loss
        # normal validation
        if normal_val_loader:
            if epoch > 0 and epoch % normal_val_interval == 0:
                val_metrics = eval(model, normal_val_metrics, normal_val_loader, device, post_transform)
                for metric in normal_val_metrics:
                    log_dict[f'n_{metric}'] = val_metrics[metric]
        # ensemble validation
        if ensemble_val_loader:
            if epoch > 0 and epoch % ensemble_val_interval == 0:
                val_metrics = eval(model, ensemble_val_metrics, ensemble_val_loader, device, post_transform)
                for metric in ensemble_val_metrics:
                    log_dict[f'en_{metric}'] = val_metrics[metric]
        if epoch > 0 and epoch % results_interval == 0:
            # visualize_results(epoch, model, normal_val_loader, device, results_dir)
            visualize_results(epoch, model, ensemble_val_loader, device, results_dir)
        # Save results and model checkpoints at the specified interval
        if epoch > 0 and epoch % checkpoint_interval == 0:
            # visualize_results(epoch, model, test_loader, device, results_dir)
            csv_files = [csv_file]
            save_checkpoint(model, optimizer, epoch, checkpoint_dir, model_name, csv_files)
        save_to_csv(csv_file, log_dict)
        log_non_none_items(logger, log_dict)
        # logger.info(f"Epoch {epoch}: Train Loss: {log_dict['train_loss']:.4f}, Val Loss: {log_dict['normal_val_loss']:.4f}, Ensemble Val Loss: {log_dict['ensemble_val_loss']:.4f}")
        # reset log_dict
        log_dict = {k: None for k in log_dict.keys()}

def log_non_none_items(logger, log_dict, idx=0):
    log_message = f"Epoch {log_dict['epoch']} - Post-transform {idx}: "
    log_message += ', '.join([f"{k}: {v:.4f}" for k, v in log_dict.items() if v is not None and k != 'epoch'])
    logger.info(log_message)

def evaluate(
        diffusion,
        model, 
        epoch,
        normal_val_loader=False,
        ensemble_val_loader=False,
        ensemble_val_metrics=[],
        device='cpu', 
        log_dir='./logs/',
        model_name='model',
        results_dir='./results',
        n_samples=4,
        ensemble_sizes=None,
        post_transforms=None):
        
    # Set model to evaluation mode
    model.eval()
    
    # Create the results directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize the logger
    log_file = os.path.join(log_dir, 'evaluation.log')
    logger = setup_logging(log_file)
    
    # Initialize log dictionaries and CSV files for each post_transform
    log_dicts = []
    csv_files = []
    
    for idx, post_transform in enumerate(post_transforms):
        log_dict = {}
        if normal_val_loader:
            log_dict['n_val_loss'] = None
        if ensemble_val_loader:
            for metric in ensemble_val_metrics:
                log_dict[f'en_{metric}'] = None
            # Initialize additional ensemble sizes
            if ensemble_sizes:
                for size in ensemble_sizes:
                    for metric in ensemble_val_metrics:
                        log_dict[f'en_{metric}_{size}'] = None
        log_dicts.append(log_dict)
        
        csv_file_path = os.path.join(log_dir, f"log_{idx}.csv")
        csv_files.append(csv_file_path)
        initialize_csv(csv_file_path, log_dict.keys())
    
    # Evaluation loop (assuming single epoch evaluation)
    with torch.no_grad():
        save_ensemble_sample_images(diffusion, ensemble_val_loader, results_dir, epoch, device, num_samples=4, ensemble_samples=n_samples, post_transforms=post_transforms)
        # Normal validation
        if normal_val_loader:
            val_loss = normal_eval(diffusion, model, normal_val_loader, device)
            for log_dict in log_dicts:
                log_dict['n_val_loss'] = val_loss
        
        # Ensemble validation
        if ensemble_val_loader:
            val_metrics_list = ensemble_eval(
                ensemble_val_metrics, diffusion, model, n_samples=n_samples, val_loader=ensemble_val_loader, 
                post_transforms=post_transforms, device=device, ensemble_sizes=ensemble_sizes)
            
            for idx, val_metrics in enumerate(val_metrics_list):
                log_dict = log_dicts[idx]
                for metric in ensemble_val_metrics:
                    log_dict[f'en_{metric}'] = val_metrics[metric]
                    # Log additional ensemble sizes if specified
                    if ensemble_sizes:
                        for size in ensemble_sizes:
                            log_dict[f'en_{metric}_{size}'] = val_metrics[f'{metric}_{size}']
        
        # Save logs and CSV files for each post_transform
        for idx, (log_dict, csv_file_path) in enumerate(zip(log_dicts, csv_files)):
            save_to_csv(csv_file_path, log_dict)
        
        # Reset log dictionaries
        log_dicts = [{k: None for k in log_dict.keys()} for log_dict in log_dicts]
