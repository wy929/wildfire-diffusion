import sys
import os
import argparse
# Parse the config path from command-line arguments
parser = argparse.ArgumentParser(description='Train a diffusion model with UNet.')
parser.add_argument('--config_path', type=str, required=True, help='Path to the config file.')
args = parser.parse_args()
# config
# config_path = './configs/train/train_64_07_ddim.yml'
# config_path = './configs/train/64_ddim_steps/train_64_01_500_ddim_20.yml'
# end config

# Use the provided config path
config_path = args.config_path
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from tqdm import tqdm
print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)
import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)

from jinja2 import Environment, FileSystemLoader
from torch import optim
from torch.utils.data import DataLoader
import torch

from src.utils.tools import read_yaml, load_and_render_config
from src.models import unet, unet_basic, autoencoder, residual_autoencoder
from src.diffusion.diffusion import Diffusion
from src.utils.transforms import post_transform, post_transform_1
from src.utils.scripts import train
from src.models.unet import UNet
from src.diffusion.diffusion import Diffusion
from src.data.dataset import FireDataset, EnsembleFireDataset
from src.utils.transforms import transform

# Automatically determine config directory from config_path
config_dir = os.path.dirname(os.path.abspath(config_path))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

env = Environment(loader=FileSystemLoader(searchpath=[
    "./",
    config_dir,
    project_root
]))
data = read_yaml(config_path)
context = {
    'config': data['config'],
    'root_dir': data['root_dir']
}
# Extract just the filename for template loading
config_filename = os.path.basename(config_path)
config = load_and_render_config(config_filename, context, env)
root_dir = config['root_dir']
os.makedirs(root_dir, exist_ok=True)
device = config['train']['device']
model_params = config['model']['parameters']
diffusion_params = config['diffusion']
# load model
model = unet.UNet(
    in_channels=model_params['in_channels'],
    model_channels=model_params['model_channels'],
    out_channels=model_params['out_channels'],
    num_res_blocks=model_params['num_res_blocks'],
    attention_resolutions=model_params['attention_resolutions'],
    dropout=model_params['dropout'],
    channel_mult=model_params['channel_mult'],
    conv_resample=model_params['conv_resample'],
    num_heads=model_params['num_heads'],
).to(device)
# load diffusion
from src.utils.transforms import post_process
diffusion = Diffusion(
    model=model,
    T=diffusion_params['T'],
    beta_start=diffusion_params['beta_start'],
    beta_end=diffusion_params['beta_end'],
    img_size=diffusion_params['img_size'],
    loss_type=diffusion_params['loss_type'],
    device=config['train']['device'],
    steps=diffusion_params['steps'],
    post_process=post_process
)
# load dataset
num_frames = config['data']['num_frames']
train_dataset_dir = config['data']['train_dataset_dir']
train_indices = config['data']['train_indices']
if config['data']['train_indices_range']:
    l, r = config['data']['train_indices_range']
    train_indices = list(range(l, r + 1))
train_batch_size = config['data']['train_batch_size']
val_dataset_dir = config['data']['val_dataset_dir']
val_indices = config['data']['val_indices']
if config['data']['val_indices_range']:
    l, r = config['data']['val_indices_range']
    val_indices = list(range(l, r + 1))
val_batch_size = config['data']['val_batch_size']
ensemble_dataset_dir = config['data']['ensemble_dataset_dir']
ensemble_indices = config['data']['ensemble_indices']
if config['data']['ensemble_indices_range']:
    l, r = config['data']['ensemble_indices_range']
    ensemble_indices = list(range(l, r + 1))
ensemble_batch_size = config['data']['ensemble_batch_size']
val_loader = None
ensemble_loader = None
print("train_indices:", train_indices)
print("val_indices:", val_indices)
print("ensemble_indices:", ensemble_indices)

# train dataset
train_dataset = FireDataset(root_dir=train_dataset_dir, 
                            indices=train_indices, 
                            num_frames=num_frames, 
                            transform=transform)
train_loader = DataLoader(train_dataset, 
                          batch_size=train_batch_size, 
                          shuffle=True)
# normal val dataset
if config['val']['normal_val']:
    val_dataset = FireDataset(root_dir=val_dataset_dir, 
                            indices=val_indices, 
                            num_frames=num_frames, 
                            transform=transform)
    val_loader = DataLoader(val_dataset, 
                            batch_size=val_batch_size, 
                            shuffle=False)
# ensemble val dataset
if config['val']['ensemble_val']:
    ensemble_dataset = EnsembleFireDataset(root_dir=ensemble_dataset_dir, 
                                        indices=ensemble_indices, 
                                        transform=transform)
    ensemble_loader = DataLoader(ensemble_dataset, 
                                batch_size=ensemble_batch_size, 
                                shuffle=False)

# optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=config['train']['lr'], momentum=0.9, weight_decay=1e-4)

pt0 = post_transform(threshold=0.02)
# pt2 = post_transform(threshold=0.05)
# pt3 = post_transform(threshold=0.1)
# pt4 = post_transform(threshold=0.2)
# pt5 = post_transform(threshold=0.3)
pt1 = post_transform_1(th1=0.02, th2=0.9)
# pt7 = post_transform_1(th1=0.05, th2=0.9)
# pt8 = post_transform_1(th1=0.1, th2=0.9)
# pt9 = post_transform_1(th1=0.2, th2=0.9)
# pt10 = post_transform_1(th1=0.3, th2=0.9)
post_transforms = [pt0, pt1]
# post_transforms = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10]
# criterion = EdgeWeightedLoss(loss_type=diffusion_params['loss_type'], edge_weight=10.0)
criterion = False # use the default loss in diffusion.py
train(
    diffusion=diffusion,
    model=model, 
    train_loader=train_loader, 
    optimizer=optimizer, 
    n_epochs=config['train']['n_epochs'], 
    loss_type=diffusion_params['loss_type'], 
    criterion=criterion,
    device=device, 
    log_dir=config['train']['log_dir'], 
    checkpoint=config['train']['checkpoint'],   
    checkpoint_dir=config['train']['checkpoint_dir'],
    checkpoint_interval=config['train']['checkpoint_interval'],
    model_name=config['model']['name'],
    results_dir=config['train']['results_dir'],
    results_interval=config['train']['results_interval'],
    patience=config['train']['patience'], 
    normal_val_loader=val_loader,
    normal_val_interval=config['val']['normal_val_interval'],
    ensemble_val_loader=ensemble_loader,
    ensemble_val_interval=config['val']['ensemble_val_interval'],
    ensemble_val_metrics=config['val']['ensemble_val_metrics'],
    n_samples=config['val']['n_samples'],
    ensemble_sizes=config['val']['ensemble_sizes'],
    post_transforms=post_transforms,
    config=config
)