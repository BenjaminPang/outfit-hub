import yaml
import os


class DatasetManager:
    def __init__(self, config_path="outfit_hub/registry.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_global_id(self, dataset_name, local_id, id_type='item'):
        """
        Convert local dataset ID to a unique global integer index.
        id_type: 'item', 'outfit', or 'user'
        """
        if dataset_name not in self.config:
            raise ValueError(f"Dataset {dataset_name} not found in registry.")
        
        offset = self.config[dataset_name][f"{id_type}_offset"]
        return int(local_id) + offset