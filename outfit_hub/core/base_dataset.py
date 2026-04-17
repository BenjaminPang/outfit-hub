#outfit_hub/core/base_dataset.py
import os
import json
from typing import Union

from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.vector_db_utils import VectorDB
from .datatypes import FashionItem


class BaseOutfitDataset(Dataset):
    def __init__(self, root_dir, dataset_name, dataset_idx=0, split='train', task_name="", encode_fn=None, encode_name="", transform=None, force_recompute=False):
        with open(os.path.join(root_dir, 'metadata.json'), 'r') as f:
            self.dataset_config = json.load(f)[dataset_name]
            self.dataset_name = dataset_name
            self.dataset_idx = dataset_idx
            self.supported_tasks = []
            for k, v in self.dataset_config['supported_tasks'].items():
                self.supported_tasks.extend([k + "_" + task for task in v.keys()])
    
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        
        # 加载标准表
        self.items_df = pd.read_parquet(os.path.join(self.dataset_dir, "items.parquet"))
        self.outfits_df = pd.read_parquet(os.path.join(self.dataset_dir, "outfits.parquet"))
        self._categories = self.items_df['category'].tolist()
        self.active_categories = list(set(self._categories))
        self._descriptions = self.items_df['description'].tolist()

        collection_name = f"{dataset_name}__{encode_name}"
        self.vector_db = VectorDB(
            self.items_df,
            collection_name,
            encode_fn,
            self.root_dir,
            persistent=True,
            force_recompute=force_recompute
        )
        
        # 筛选 Split
        if split in self.outfits_df.columns or 'split' in self.outfits_df.columns:
            self.outfits_df = self.outfits_df[self.outfits_df['split'] == split].reset_index(drop=True)

        # 图片处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        

    def get_feature(self, item_idx: Union[int, list[int]]) -> np.ndarray:
        return self.vector_db._embedding_cache[item_idx]
        
    def construct_item(self, iidx: int) -> FashionItem:
        try:
            return FashionItem(
                item_idx=iidx,
                category=self._categories[iidx],
                image=None,
                description= self._descriptions[iidx],
                embedding=self.get_feature(iidx)
            )
        except IndexError:
            return None
    
    def get_image(self, item_idx, return_tensor=True):
        """
        Priority 1: Load from 'images/' directory (Fastest)
        Priority 2: Fallback to Tar archives (If images/ folder is missing)
        """
        img_path = os.path.join(self.dataset_dir, 'images', f"{item_idx}.jpg")
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                return self.transform(img) if return_tensor else img
            except Exception as e:
                print(f"⚠️ Warning: Failed to load {img_path}, error: {e}")
                return None

    def __len__(self):
        return len(self.outfits_df)
