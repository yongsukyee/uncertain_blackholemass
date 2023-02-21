# GET CONFIGS

import yaml

def get_config(config_path='config/config.yaml'):
    """Return configs"""
    print(f"Load config file >> {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
