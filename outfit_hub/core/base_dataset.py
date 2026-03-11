#outfit_hub/core/base_dataset.py
import os
import tarfile
import io
import json
import pickle

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.vector_db_utils import VectorDB


class BaseOutfitDataset(Dataset):
    def __init__(self, root_dir, dataset_name, split='train', transform=None):
        with open(os.path.join(root_dir, 'metadata.json'), 'r') as f:
            self.dataset_config = json.load(f)[dataset_name]
            self.dataset_name = dataset_name
            self.chunk_size = self.dataset_config.get('chunk_size', 50000)
            self.supported_tasks = []
            for k, v in self.dataset_config['supported_tasks'].items():
                self.supported_tasks.extend([k + "_" + task for task in v.keys()])
    
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        
        # 加载标准表
        self.items_df = pd.read_parquet(os.path.join(self.dataset_dir, "items.parquet"))
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
        
        # 句柄缓存 (Lazy Loading)
        self.tar_handles = {}

    @property
    def vector_db(self):
        if self._vector_db is None:
            print("🔍 Initializing VectorDB on demand...")
            # 执行加载逻辑
            clip_features_path = os.path.join(self.dataset_dir, 'clip_vision_features.pkl')
            with open(clip_features_path, 'rb') as f:
                self.clip_features = pickle.load(f)  # list type
            self._vector_db = VectorDB(self.items_df, self.clip_features, self.dataset_name, self.root_dir)
        return self._vector_db

    def _get_tar_path(self, item_idx):
        """根据 item_idx 计算 tar 文件的绝对路径"""
        tar_idx = item_idx // self.chunk_size
        return os.path.join(self.dataset_dir, f"{tar_idx:03d}.tar")

    def load_img_from_tar(self, item_idx, return_tensor=True):
        """从 Tar 包读取单张图片"""
        path = self._get_tar_path(item_idx)
        
        if path not in self.tar_handles:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Tar not found: {path}")
            self.tar_handles[path] = tarfile.open(path, "r")
            
        try:
            tar = self.tar_handles[path]
            member_name = f"{item_idx}.jpg"
            member = tar.getmember(member_name)
            f = tar.extractfile(member)
            img = Image.open(io.BytesIO(f.read())).convert('RGB')
            if return_tensor:
                return self.transform(img)
            else:
                return img
        except Exception as e:
            # 如果读取失败，返回全黑图（建议在训练时排查此类问题）
            print(f"Error loading image {item_idx}: {e}")
            return self._error_placeholder(return_tensor)

    def __len__(self):
        return len(self.outfits_df)


