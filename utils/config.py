import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Loads the YAML configuration file.
    
    Args:
        config_path (str): Path to the config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def get_experiment_config(experiment_name, base_config_path="config.yaml"):
    """
    Utility to potentially merge experiment specific overrides.
    """
    config = load_config(base_config_path)
    # Add experiment specific modifications here if needed
    config['experiment_name'] = experiment_name
    return config
