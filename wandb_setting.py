from typing import Callable
import wandb
from omegaconf import OmegaConf, DictConfig

def wandb_setting(
    entity:str,
    project:str,
    group_name:str,
    experiment_name:str,
    arg_config:str) -> Callable:
    
    wandb.login()
    config = OmegaConf.load(f'./configs/{arg_config}.yaml')
    print('='*50,OmegaConf.to_yaml(config), '='*50, sep='\n')
    assert type(config) == DictConfig
    
    wandb.init(project=project, group=group_name, name=experiment_name, entity=entity)
    wandb.config = config
    
    return wandb.config