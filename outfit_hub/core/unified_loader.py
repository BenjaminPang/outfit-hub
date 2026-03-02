# fashion_hub/core/unified_loader.py
import torch
from torch.utils.data import ConcatDataset

from ..datasets.pog import POGDataset
from ..datasets.polyvore import PolyvoreDataset
from ..datasets.hashnet import HashNetDataset


class FashionUnifiedDataset:
    """
    The unified entry point to load multiple fashion datasets.
    """
    DATASET_REGISTRY = {
        "pog": POGDataset,
        "polyvore": PolyvoreDataset,
        "hashnet": HashNetDataset
    }

    def __init__(self, dataset_names, mode='train', task='outfit', transform=None):
        """
        Args:
            dataset_names (list or str): e.g., ["pog", "polyvore"] or "hashnet"
            mode (str): 'train', 'valid', 'test'
            task (str): 'outfit' (standard), 'fitb' (Fill-In-The-Blank), 'auc'
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
            
        self.datasets = []
        for name in dataset_names:
            if name.lower() not in self.DATASET_REGISTRY:
                raise ValueError(f"Dataset {name} not supported.")
            
            # Initialize individual dataset implementation
            ds_instance = self.DATASET_REGISTRY[name.lower()](
                mode=mode, 
                task=task, 
                transform=transform
            )
            self.datasets.append(ds_instance)

    def get_loader(self, batch_size=32, shuffle=True, **kwargs):
        """
        Returns a PyTorch DataLoader wrapping the concatenated datasets.
        """
        combined = ConcatDataset(self.datasets)
        return torch.utils.data.DataLoader(
            combined, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            **kwargs
        )