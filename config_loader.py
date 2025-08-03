"""
Configuration loader for Assignment #3
"""

import yaml
from pathlib import Path
from data_types import Config


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    # Flatten nested config
    flat_config = {}
    for section, values in config_data.items():
        flat_config.update(values)
    
    return Config(**flat_config)