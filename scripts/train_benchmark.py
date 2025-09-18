import sys
import os
import argparse
# Parse the config path from command-line arguments
parser = argparse.ArgumentParser(description='Train benchmark model.')
parser.add_argument('--config_path', type=str, required=True, help='Path to the config file.')
args = parser.parse_args()
# Use the provided config path
config_path = args.config_path
# config
# config_path = './configs/train/64_ddim_steps/train_64_01_500_bm_unet_02.yml'
# end config
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)
import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
from jinja2 import Environment, FileSystemLoader
from src.utils.tools import read_yaml, load_and_render_config
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import FireDataset, EnsembleFireDataset
from src.utils.transforms import transform, post_process
from src.models import unet, unet_basic
from src.utils.scripts import train_normal

env = Environment(loader=FileSystemLoader(searchpath="./"))
data = read_yaml(config_path)
context = {
    'config': data['config'],
    'root_dir': data['root_dir']
}
config = load_and_render_config(config_path, context, env)
root_dir = config['root_dir']
os.makedirs(root_dir, exist_ok=True)
device = config['train']['device']
model_params = config['model']['parameters']
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

print('val_loader:', val_loader)
print('ensemble_loader:', ensemble_loader)
print("train_indices:", train_indices)
print("val_indices:", val_indices)
print("ensemble_indices:", ensemble_indices)

optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'])
# optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=1e-4) # 03
criterion = nn.MSELoss()
from src.utils.transforms import ThresholdToZero
# post_transform = ThresholdToZero(threshold=0.3) 
# post_transform = ThresholdToZero(threshold=0.0) # 01
post_transform = post_process   # threshold=0.5 by default

train_normal(
          model,
          train_loader=train_loader, 
          optimizer=optimizer, 
          n_epochs=config['train']['n_epochs'], 
          criterion=criterion, 
          device=device,  
          log_file=config['train']['log_file'], 
          csv_file=config['train']['csv_file'], 
          checkpoint=config['train']['checkpoint'],
          checkpoint_dir=config['train']['checkpoint_dir'], 
          checkpoint_interval=config['train']['checkpoint_interval'],
          model_name=config['model']['name'],
          results_dir=config['train']['results_dir'],
          results_interval=config['train']['results_interval'],
          patience=5,
          normal_val_loader=val_loader,
          normal_val_interval=config['val']['normal_val_interval'],
          normal_val_metrics=config['val']['normal_val_metrics'],
          ensemble_val_loader=ensemble_loader,
          ensemble_val_interval=config['val']['ensemble_val_interval'],
          ensemble_val_metrics=config['val']['ensemble_val_metrics'],
          n_samples=config['val']['n_samples'],
          post_transform=post_transform,
          config=config)