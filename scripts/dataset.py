import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.tools import read_yaml
from src.utils.data import Generator
# path to the config file
config_name = "dataset_64_05"
# config_name = "dataset_64_04_ensemble"
# config_name = "dataset_128_02_ensemble"
config_path = f"./configs/dataset/{config_name}.yml"
# read the config file
config = read_yaml(config_path)
print(config)
# generate the dataset
generator = Generator(config_path=config['config_path'], 
                      data_dir_path=config['data_dir_path'], 
                      np_dir_path=config['np_dir_path'], 
                      sample_size=config['sample_size'],
                      frame_size=config['frame_size'],
                      generations=config['generations'],
                      max_burned_area_percentage=config['max_burned_area_percentage']
                      )
if config['type'] == 'raw':
    generator(type=config['type'], 
              num_samples=config['num_samples'], 
              interval=config['interval'],
              fire_num=config['fire_num'], 
              tolerance=config['tolerance'],
              info=config['info'])
elif config['type'] == 'ensemble':
    generator(type=config['type'], 
              num_samples=config['num_samples'],                  
              fire_num=config['fire_num'], 
              num_updates=config['num_updates'],
              frames=config['frames'],
              tolerance=config['tolerance'],
              info=config['info'])