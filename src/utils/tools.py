import torch
import datetime
import os
import logging
import csv
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from src.utils.transforms import transform
import numpy as np
from torchvision.utils import save_image, make_grid
import pandas as pd
import yaml


# Function to read a YAML file
def read_yaml(file_path):
    """
    Reads a YAML file and returns the content as a dictionary.

    :param file_path: Path to the YAML file
    :return: Dictionary containing the YAML file content
    """
    with open(file_path, 'r') as file:
        try:
            content = yaml.safe_load(file)
            return content
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return None

def load_and_render_config(template_path, context, env):
    template = env.get_template(template_path)
    rendered_content = template.render(context)
    # set yaml loader
    config = yaml.safe_load(rendered_content)
    return config

# Function to scale image pixel values
def scale_255_to_0_1(imgs):
    """
    Scale image pixel values from [0, 255] to [0, 1].
    
    Parameters:
    - img (numpy.ndarray or torch.Tensor): Image array with pixel values in the range [0, 255].

    Returns:
    - numpy.ndarray or torch.Tensor: Image array with pixel values scaled to the range [0, 1].
    """
    if isinstance(imgs, torch.Tensor):
        return imgs.float() / 255.0
    elif isinstance(imgs, np.ndarray):
        return imgs.astype(np.float32) / 255.0
    else:
        raise TypeError("Input should be a numpy.ndarray or torch.Tensor.")

def scale_0_1_to_255(imgs):
    """
    Scale image pixel values from [0, 1] to [0, 255].
    
    Parameters:
    - img (numpy.ndarray or torch.Tensor): Image array with pixel values in the range [0, 1].

    Returns:
    - numpy.ndarray or torch.Tensor: Image array with pixel values scaled to the range [0, 255].
    """
    if isinstance(imgs, torch.Tensor):
        return (imgs * 255).byte()
    elif isinstance(imgs, np.ndarray):
        return (imgs * 255).astype(np.uint8)
    else:
        raise TypeError("Input should be a numpy.ndarray or torch.Tensor.")

def save_checkpoint(model, optimizer, epoch, path, model_name, log_csvs=None, best=False):
    # mkdir if not exists
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if log_csvs:
        for idx, log_csv in enumerate(log_csvs):
            log_df = pd.read_csv(log_csv)
            checkpoint[f'log_{idx}'] = log_df
    if best==False:
        model_name = f"{model_name}_{epoch}.pth"
    else:
        model_name = f"{model_name}_best.pth"
    checkpoint_path = os.path.join(path, model_name)
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

# Function to setup logging
def setup_logging(log_file='./training.log'):
    # mkdir if not exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # check and backup existing log file
    if os.path.exists(log_file):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_log_file = f"{log_file}.{timestamp}.bak"
        os.rename(log_file, backup_log_file)
    
    # clear existing handlers
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # set logging level
    logger.setLevel(logging.INFO)
    
    # create file handler
    handler = logging.FileHandler(log_file, mode='w')  # use 'w' to overwrite existing log file
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))  # set format
    logger.addHandler(handler)

    return logger

def initialize_csv(csv_file, fieldnames):
    # mkdir if not exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # check and backup existing CSV file
    if os.path.exists(csv_file):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_csv_file = f"{csv_file}.{timestamp}.bak"
        os.rename(csv_file, backup_csv_file)
    
    # clear existing CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

# def save_to_csv(csv_file, data_dict):
#     with open(csv_file, 'a', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=data_dict.keys())
#         writer.writerow(data_dict)

def save_to_csv(csv_file, log_dict, decimal_places=4):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"{v:.{decimal_places}f}" if isinstance(v, float) else v for v in log_dict.values()])

# def save_to_csv(csv_file, log_dict):
#     import csv
#     with open(csv_file, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(log_dict.values())

# Function to save the image

def save_imgs(imgs, save_dir, img_name, img_format='png', scale=False):
    """
    Save a batch of predicted images as a single composite grayscale image.

    Parameters:
    - pred_imgs (torch.Tensor): Batch of predicted images (tensor).
    - save_dir (str): Directory where the composite image will be saved.
    - img_name (str): Name of the composite image file.
    - img_format (str): Format to save the image, e.g., 'png', 'jpg'.
    - scale (bool): Whether to scale the pixel values to 8-bit (0-255) before saving.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Convert the batch of images to a list of PIL grayscale images
    pil_images = []
    for i in range(imgs.shape[0]):
        img = imgs[i].cpu().detach().numpy()  # Convert to numpy array
        if scale:
            # img = (img * 255).astype('uint8')  # Convert to 8-bit image
            img = scale_0_1_to_255(img)
        # Assuming the predicted images have a single channel (grayscale)
        if img.shape[0] == 1:
            img = img.squeeze(0)  # Remove the channel dimension
        else:
            img = img.mean(axis=0)  # Convert multi-channel to single channel by averaging

        # Convert numpy array to PIL image
        img_pil = Image.fromarray(img, mode='L')
        pil_images.append(img_pil)

    # Get the dimensions of the first image
    img_width, img_height = pil_images[0].size

    # Create a new blank image with the appropriate size to hold all images in a row
    composite_image = Image.new('L', (img_width * len(pil_images), img_height))

    # Paste each image into the composite image
    for idx, img_pil in enumerate(pil_images):
        composite_image.paste(img_pil, (idx * img_width, 0))

    # Save the composite image
    save_path = os.path.join(save_dir, f'{img_name}.{img_format}')
    composite_image.save(save_path, format=img_format)

# Redefine the function with the import correction
def process_and_save_ensemble_images(root_dir, transform=None, info=False):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            org_image_path = os.path.join(folder_path, 'org.png')
            update_dir = os.path.join(folder_path, 'update')

            processed_images = []
            for img_name in os.listdir(update_dir):
                img_path = os.path.join(update_dir, img_name)
                image = Image.open(img_path).convert('L')
                if transform is None:
                    transform = transforms.Compose([transforms.ToTensor()])
                processed_image = transform(image)
                processed_images.append(processed_image.numpy())

            # Create ensemble image by averaging
            ensemble_image = np.mean(processed_images, axis=0)
            
            # Save the ensemble image
            ensemble_image_path = os.path.join(folder_path, 'ensemble.png')
            ensemble_pil_image = transforms.ToPILImage()(torch.tensor(ensemble_image))
            ensemble_pil_image.save(ensemble_image_path)

            # Display the original, processed, and ensemble images
            if info:
                fig, axes = plt.subplots(1, len(processed_images) + 2, figsize=(20, 10))
                axes[0].imshow(Image.open(org_image_path).convert('L'), cmap='gray')
                axes[0].set_title('Original')

                for i, img in enumerate(processed_images):
                    axes[i + 1].imshow(img[0], cmap='gray')
                    axes[i + 1].set_title(f'Processed {i+1}')

                axes[-1].imshow(ensemble_image[0], cmap='gray')
                axes[-1].set_title('Ensemble')

                plt.show()
# Function to display images
def show_imgs(imgs, cmap='gray'):
    if isinstance(imgs, torch.Tensor):
        if imgs.is_cuda:
            imgs = imgs.cpu()
        imgs = imgs.numpy()
    
    if isinstance(imgs, np.ndarray):
        if len(imgs.shape) == 4:  # batch of images [b, c, h, w]
            b, c, h, w = imgs.shape
            imgs = imgs.transpose(0, 2, 3, 1)  # convert to [b, h, w, c]
        elif len(imgs.shape) == 3:  # single image with channel [c, h, w]
            c, h, w = imgs.shape
            imgs = imgs.transpose(1, 2, 0)  # convert to [h, w, c]
        elif len(imgs.shape) == 2:  # single grayscale image [h, w]
            pass
        else:
            raise ValueError("Unsupported shape: {}".format(imgs.shape))
    else:
        raise TypeError("Input should be a torch.Tensor or np.ndarray")
    
    if len(imgs.shape) == 3 and imgs.shape[2] == 1:  # single grayscale image [h, w, 1]
        imgs = imgs.squeeze(2)
    
    if len(imgs.shape) == 4:  # batch of images [b, h, w, c] or [b, h, w]
        num_imgs = imgs.shape[0]
        fig, axes = plt.subplots(1, num_imgs, figsize=(num_imgs * 3, 3))
        for i in range(num_imgs):
            axes[i].imshow(imgs[i], cmap=cmap)
            axes[i].axis('off')
    elif len(imgs.shape) == 3:  # single image [h, w, c]
        plt.imshow(imgs, cmap=cmap)
        plt.axis('off')
    elif len(imgs.shape) == 2:  # single grayscale image [h, w]
        plt.imshow(imgs, cmap=cmap)
        plt.axis('off')
    else:
        raise ValueError("Unsupported shape after processing: {}".format(imgs.shape))
    
    plt.show()