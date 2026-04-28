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

from .datatypes import FashionItem, FashionOutfit


class BaseOutfitDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        dataset_name: str,
        feature_path: str, # 只需要传入 npy 文件的绝对路径
        split: str = 'train',
        transform: str = None,
        **kwargs
    ):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        self.feature_path = feature_path
        self.dataset_idx = kwargs.get("dataset_idx", 0)
        
        # 加载标准表
        items_df = pd.read_parquet(os.path.join(self.dataset_dir, "items.parquet"))
        self.items_number = len(items_df)
        self.cat_to_indices = items_df.groupby('category').groups
        self._categories = items_df['category'].tolist()
        self.active_categories = list(set(self._categories))
        self._descriptions = items_df['description'].fillna('').tolist()

        outfits_df = pd.read_parquet(os.path.join(self.dataset_dir, "outfits.parquet"))
        if split == "all":
            self.outfits_df = outfits_df
        else:
            self.outfits_df = outfits_df[outfits_df['split'] == split].reset_index(drop=True)
        
        self._outfits = None
        self._features = None

        # 图片处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def outfits(self):
        if self._outfits is None:
            self._outfits = [
                json.loads(x) if isinstance(x, str) else x 
                for x in self.outfits_df['item_indices'].tolist() 
            ]
            self.outfits_df = None
        return self._outfits

    @property
    def features(self):
        """
        延迟加载 mmap。
        """
        if self._features is None and self.feature_path:
            if os.path.exists(self.feature_path):
                # 'r' 模式非常关键：它保证了多进程共享物理内存，且是只读的
                self._features = np.load(self.feature_path, mmap_mode='r')
            else:
                print(f"⚠️ Warning: feature_path {self.feature_path} not found.")
        return self._features
    
    def get_feature(self, item_idx: Union[int, list[int]]) -> np.ndarray:
        """
        直接通过 numpy 索引获取特征
        """
        if self.features is not None:
            # numpy mmap 对象支持标准切片和索引
            return self.features[item_idx]
        return None
        
    def construct_item(self, iidx: int, include_image=False) -> FashionItem:
        try:
            return FashionItem(
                item_idx=iidx,
                category=self._categories[iidx],
                image=self.get_image(iidx, return_tensor=False) if include_image else None,
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
        return len(self.outfits)
    
    def __getitem__(self, idx):
        """返回一条完整的 Outfit 数据"""
        item_ids = self.outfits[idx]
        
        # 构造这一组搭配的所有 FashionItem
        items = [self.construct_item(iid) for iid in item_ids]
        
        return FashionOutfit(
            outfit=items
        )
