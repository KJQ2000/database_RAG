from pathlib import Path
import yaml

def load_config(file_path:Path):

    # Open the YAML config file and load it
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config