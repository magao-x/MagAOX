import yaml
import os


def parse_config_file(config_file: str) -> dict:
    """Parse the configuration file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")
    with open(config_file, 'r') as yaml_file:
        config_params = yaml.safe_load(yaml_file)
    return config_params