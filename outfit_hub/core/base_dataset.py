#outfit_hub/core/base_dataset.py
import os
import json

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.vector_db_utils import VectorDB


class BaseOutfitDataset(Dataset):
    def __init__(self, root_dir, dataset_name, dataset_idx=0, split='train', encode_fn=None, encode_name="", max_seq_length=9, transform=None, force_recompute=False):
        with open(os.path.join(root_dir, 'metadata.json'), 'r') as f:
            self.dataset_config = json.load(f)[dataset_name]
            self.dataset_name = dataset_name
            self.dataset_idx = dataset_idx
            self.max_seq_length = max_seq_length
            self.supported_tasks = []
            for k, v in self.dataset_config['supported_tasks'].items():
                self.supported_tasks.extend([k + "_" + task for task in v.keys()])
    
        self.root_dir = root_dir
        self.dataset_dir = os.path.join(root_dir, dataset_name)
        self.split = split
        
        # 加载标准表
        self.items_df = pd.read_parquet(os.path.join(self.dataset_dir, "items.parquet"))
        self.num_items = len(self.items_df)
        self.outfits_df = pd.read_parquet(os.path.join(self.dataset_dir, "outfits.parquet"))
        collection_name = f"{dataset_name}__{encode_name}"
        self.vector_db = VectorDB(self.items_df, collection_name, self.root_dir, persistent=True)
        self._embedding_cache = None
        self.encode_fn = encode_fn
            
        # 筛选 Split
        if split in self.outfits_df.columns or 'split' in self.outfits_df.columns:
            self.outfits_df = self.outfits_df[self.outfits_df['split'] == split].reset_index(drop=True)

        # 图片处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sync_embeddings(force_recompute)

    def _load_vector_db_to_numpy(self):
        res = self.vector_db.collection.get(include=['embeddings'])
        
        dim = len(res['embeddings'][0])
        all_embs = np.zeros((self.num_items, dim), dtype=np.float32)
        
        ids = np.array(res['ids'], dtype=int)
        embs = np.array(res['embeddings'], dtype=np.float32)
        
        all_embs[ids] = embs 
        return all_embs
    
    def get_feature(self, item_idx: int) -> np.ndarray:
        if self._embedding_cache is not None:
            return self._embedding_cache[item_idx]
        else:
            result = self.vector_db.collection.get(str(item_idx), include=['embeddings'])
            if result['embeddings'] is not None:
                return np.array(result['embeddings'][0], dtype=np.float32)
            else:
                raise KeyError(f"Item ID {item_idx} not found in VectorDB. Please run sync_embeddings first.")
    
    def sync_embeddings(self, force_recompute=False):
        """
        Core Management Method:
            - If `force_recompute=True`, clear the current Collection and recompute.
            - Otherwise, only fill in the missing features.
        """
        batch_size=5000
        if force_recompute or self.vector_db.collection.count() < len(self.items_df):
            if self.encode_fn is None:
                raise ValueError("Need encode_fn to sync embeddings.")

            print(f"⚠️ Warning: Recomputing ALL embeddings for {self.vector_db.collection_name}")
            self.vector_db.clear_collection() 

            all_idxs = self.items_df.index.tolist()
            if all_idxs:
                for i in tqdm(range(0, len(all_idxs), batch_size), desc="Syncing DB"):
                    batch_idxs = all_idxs[i : i + batch_size]

                    imgs = [self.get_image(idx, return_tensor=False) for idx in batch_idxs]
                    txts = self.items_df.loc[batch_idxs, 'description'].fillna('').tolist()
                    
                    embs = self.encode_fn(imgs, txts)
                    metadatas = self.items_df.loc[batch_idxs].to_dict(orient='records')
                    self.vector_db.update_features(batch_idxs, embs, metadatas)
        
        # self._load_to_ram()

    # def _load_to_ram(self):
    #     """将 DB 数据一次性拉入内存，确保评测时的极速"""
    #     res = self.vector_db.collection.get(include=['embeddings'])
    #     dim = len(res['embeddings'][0])
    #     # 创建 [num_items, dim] 的矩阵，注意 ID 映射
    #     self._embedding_cache = np.zeros((self.items_df.index.max() + 1, dim), dtype=np.float32)
    #     for i, idx_str in enumerate(res['ids']):
    #         self._embedding_cache[int(idx_str)] = res['embeddings'][i]
    
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


