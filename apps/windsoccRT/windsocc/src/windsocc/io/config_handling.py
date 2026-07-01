import yaml
import os


def parse_config_file(config_file: str) -> dict:
    """Parse the configuration file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")
    with open(config_file, 'r') as yaml_file:
        config_params = yaml.safe_load(yaml_file)
    return config_params

def resolve_config_path(path_arg, explicit_config=None):
    """
    Resolve the working directory and config file path.

    The command accepts either a directory containing `ws_config.yaml`, or a
    path to the YAML file itself.
    """
    config_candidate = explicit_config if explicit_config is not None else path_arg
    resolved_path = os.path.abspath(config_candidate)

    if os.path.isdir(resolved_path):
        return resolved_path, os.path.join(resolved_path, "ws_config.yaml")

    return os.path.dirname(resolved_path), resolved_path