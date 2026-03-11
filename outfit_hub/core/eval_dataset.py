#outfit_hub/core/eval_dataset.py
import os
import json
import torch
from torch.utils.data import DataLoader

from .base_dataset import BaseOutfitDataset


class FITBEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir, dataset_name, task_name, split='test', **kwargs):
        print(f"Loading FITB Data for evaluating...")
        super().__init__(root_dir, dataset_name, split, **kwargs)
        with open(os.path.join(self.dataset_dir, "eval", f"{task_name}_{split}.json"), 'r') as f:
            self.tasks = json.load(f)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        return task


class CompEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir, dataset_name, task_name, split='test', **kwargs):
        print(f"Loading Compatibility Data for evaluating...")
        super().__init__(root_dir, dataset_name, split, **kwargs)
        with open(os.path.join(self.dataset_dir, "eval", f"{task_name}_{split}.json"), 'r') as f:
            self.tasks = json.load(f)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        return task
