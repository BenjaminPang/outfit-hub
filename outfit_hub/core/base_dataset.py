#outfit_hub/core/base_dataset.py
import os
import json

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.vector_db_utils import VectorDB


class BaseOutfitDataset(Dataset):
    def __init__(self, root_dir, dataset_name, dataset_idx=0, split='train', max_seq_length=9, transform=None, load_clip=True, load_img=False):
        with open(os.path.join(root_dir, 'metadata.json'), 'r') as f:
            self.dataset_config = json.load(f)[dataset_name]
            self.dataset_name = dataset_name
            self.dataset_idx = dataset_idx
            self.max_seq_length = max_seq_length
            self.chunk_size = self.dataset_config.get('chunk_size', 50000)
            self.supported_tasks = []
            for k, v in self.dataset_config['supported_tasks'].items():
                self.supported_tasks.extend([k + "_" + task for task in v.keys()])
    
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        self.load_clip = load_clip
        self.load_img = load_img
        
        # 加载标准表
        self.items_df = pd.read_parquet(os.path.join(self.dataset_dir, "items.parquet"))
        self.num_items = len(self.items_df)
        self.outfits_df = pd.read_parquet(os.path.join(self.dataset_dir, "outfits.parquet"))
        self._vector_db = None
            
        # 筛选 Split
        if split in self.outfits_df.columns or 'split' in self.outfits_df.columns:
            self.outfits_df = self.outfits_df[self.outfits_df['split'] == split].reset_index(drop=True)

        # 图片处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.clip_feature_path = os.path.join(self.dataset_dir, 'clip_vision_features.npy')
        self._clip_features = np.memmap(
            self.clip_feature_path, 
            dtype='float32', 
            mode='r', 
            shape=(self.num_items, 512)
        )

    @property
    def vector_db(self):
        if self._vector_db is None:
            print("🔍 Initializing VectorDB on demand...")
            self._vector_db = VectorDB(self.items_df, self._clip_features, self.dataset_name, self.root_dir)
        return self._vector_db
    
    def get_clip_feature(self, item_idx):
        feat = self._clip_features[item_idx]
        if isinstance(feat, np.ndarray):
            feat_writable = feat.copy() 
            return torch.from_numpy(feat_writable).float()
        else:
            return torch.tensor(feat).float()
    
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


