import os
import tarfile
import io
import json
import pickle

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseOutfitDataset(Dataset):
    def __init__(self, root_dir, dataset_name, split='train', transform=None, load_img=False, load_clip=False):
        self.root_dir = os.path.join(root_dir, dataset_name)
        
        with open(os.path.join(self.root_dir, 'metadata.json'), 'r') as f:
            self.dataset_config = json.load(f)[dataset_name]
            self.dataset_name = dataset_name
            self.chunk_size = self.dataset_config.get('chunk_size', 50000)

        self.split = split
        self.load_img = load_img
        self.load_clip = load_clip
        
        # 1. 加载标准表
        self.items_df = pd.read_parquet(os.path.join(self.root_dir, "items.parquet"))
        self.outfits_df = pd.read_parquet(os.path.join(self.root_dir, "outfits.parquet"))
        if load_clip:
            with open(os.path.join(self.root_dir, 'clip_vision_features.pkl'), 'rb') as f:
                self.clip_features = pickle.load(f)  # list type
        else:
            self.clip_features = []
            
        
        # 筛选 Split
        if split in self.outfits_df.columns or 'split' in self.outfits_df.columns:
            self.outfits_df = self.outfits_df[self.outfits_df['split'] == split].reset_index(drop=True)

        # 2. 图片处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 3. 句柄缓存 (Lazy Loading)
        self.tar_handles = {}

    def _get_tar_path(self, item_idx):
        """根据 item_idx 计算 tar 文件的绝对路径"""
        tar_idx = item_idx // self.chunk_size
        return os.path.join(self.root_dir, f"images_{tar_idx:03d}.tar")

    def _load_img_from_tar(self, item_idx, return_tensor=True):
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
    

class OutfitTrainDataset(BaseOutfitDataset):
    def __getitem__(self, idx):
        row = self.outfits_df.iloc[idx]
        item_indices = row['item_indices'] # List of ints
        
        # 默认只返回索引，灵活性留给子类
        data = {
            "item_indices": torch.tensor(item_indices, dtype=torch.long),
            "length": len(item_indices),
            "source": self.dataset_name,
        }
        
        if self.load_img:
            data["item_imgs"] = torch.stack([self._load_img_from_tar(i) for i in item_indices])

        if self.load_clip:
            data["item_embeddings"] = torch.stack([self.clip_features[i] for i in item_indices])
            
        return data

    @staticmethod
    def collate_fn(batch):
        """通用的 Padding 逻辑"""
        from torch.nn.utils.rnn import pad_sequence
        item_idxs = [item["item_indices"] for item in batch]
        padded_idxs = pad_sequence(item_idxs, batch_first=True, padding_value=-1)
        
        # 构建 Mask: 1 为 padding, 0 为有效
        mask = (padded_idxs == -1)
        
        res = {
            "item_idxs": padded_idxs,
            "mask": mask,
            "outfit_idxs": torch.tensor([item["outfit_idx"] for item in batch])
        }

        if "item_embeddings" in batch[0]:
            embeddings = [item["item_embeddings"] for item in batch]
            res["item_embeddings"] = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        
        if "item_imgs" in batch[0]:
            imgs = [item["item_imgs"] for item in batch]
            res["item_imgs"] = pad_sequence(imgs, batch_first=True, padding_value=0.0)
            
        return res


class FITBEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir, dataset_name, split='test', **kwargs):
        super().__init__(root_dir, dataset_name, split, **kwargs)
        with open(os.path.join(self.root_dir, "eval", f"fitb_{split}.json"), 'r') as f:
            self.tasks = json.load(f)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        # 标准输出格式：上下文，候选人，GT
        return {
            "item_candidates": torch.tensor(task['item_candidates'], dtype=torch.long),
            "blank_position": task['blank_position'],
            "gt_item_idx": task['gt_item_idx'],
            "gt_label": task['gt_outfit_label'],
            "original_outfit": torch.tensor(task['original_outfit'], dtype=torch.long)
        }


class CompEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir, dataset_name, split='test', is_hard=False, **kwargs):
        super().__init__(root_dir, dataset_name, split, **kwargs)
        prefix = "compatibility_hard" if is_hard else "compatibility"
        with open(os.path.join(self.root_dir, "eval", f"{prefix}_{split}.json"), 'r') as f:
            self.tasks = json.load(f)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        return {
            "item_indices": torch.tensor(task['items'], dtype=torch.long),
            "label": task['label'],
            "user_idx": task.get('user_idx', -1)
        }