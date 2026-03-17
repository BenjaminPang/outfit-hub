import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence

from .base_dataset import BaseOutfitDataset
    

class OutfitSequenceDataset(BaseOutfitDataset):
    """
    Generic Sequence Dataset: Responsible for extracting complete outfit items and their associated features.
    Applicable Scenarios: Behavior Cloning (BC), Autoencoding, Basic Feature Extraction, etc.
    """
    def __getitem__(self, idx):
        row = self.outfits_df.iloc[idx]
        item_indices = row['item_indices']
        if isinstance(item_indices, str):
            item_indices = json.loads(item_indices)
        item_indices = [int(i) for i in item_indices] # List of ints
        if len(item_indices) > self.max_seq_length:
            if self.split == "train":
                item_indices = random.sample(item_indices, self.max_seq_length)
            else:
                item_indices = item_indices[:self.max_seq_length]
        
        # 默认只返回索引，灵活性留给子类
        data = {
            "outfit_idx": idx,
            "item_indices": torch.tensor(item_indices, dtype=torch.long),
            "length": len(item_indices),
            "dataset_idx": self.dataset_idx,
        }
        
        if self.load_img:
            data["item_imgs"] = torch.stack([self.get_image(idx) for idx in item_indices])

        if self.load_clip:
            data["item_embeddings"] = torch.stack([self.get_clip_feature(idx) for idx in item_indices])
            
        return data

    @staticmethod
    def collate_fn(batch):
        """通用的 Padding 逻辑"""
        
        item_idxs = [item["item_indices"] for item in batch]
        padded_idxs = pad_sequence(item_idxs, batch_first=True, padding_value=-1)
        
        # 构建 Mask: 1 为 padding, 0 为有效
        mask = (padded_idxs == -1)
        
        res = {
            "item_idxs": padded_idxs,
            "mask": mask,
            "dataset_idxs": torch.tensor([item["dataset_idx"] for item in batch], dtype=torch.long),
            "outfit_idxs": torch.tensor([item["outfit_idx"] for item in batch])
        }

        if "item_embeddings" in batch[0]:
            embeddings = [item["item_embeddings"] for item in batch]
            res["item_embeddings"] = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        
        if "item_imgs" in batch[0]:
            imgs = [item["item_imgs"] for item in batch]
            res["item_imgs"] = pad_sequence(imgs, batch_first=True, padding_value=0.0)
            
        return res


class OutfitValueDataset(BaseOutfitDataset):
    """
    Value Function Dataset: Used to train the Value Head (VH).
    Purpose: By constructing positive, negative, and incomplete samples, it enables the model to learn how to evaluate the state of the current combination.
    """
    def __getitem__(self, idx):
        row = self.outfits_df.iloc[idx]
        item_indices = row['item_indices']
        if isinstance(item_indices, str):
            item_indices = json.loads(item_indices)

        item_indices = item_indices[:self.max_seq_length]

        full_embeds = torch.stack([self.get_clip_feature(int(i)) for i in item_indices])
        outfit_length = full_embeds.size(0)

        mode = random.random()
        if mode < 0.5:
            # 模式 A: 正样本-已完成 (Label: [1, 1])
            embeddings = full_embeds
            label = [0.9, 0.9]
            
        elif mode < 0.7:
            # 模式 B: 正样本-未完成 (Label: [1, -1])
            # 随机截断，保留至少 1 个单品
            if outfit_length > 1:
                cut_len = random.randint(1, outfit_length - 1)
                embeddings = full_embeds[:cut_len]
            else:
                embeddings = full_embeds
            label = [0.9, -0.9]
            
        else:
            # 模式 C: 负样本-逻辑毁坏 (Label: [-1, -1])
            # 随机替换其中一个位置
            embeddings = full_embeds.clone()
            replace_pos = random.randint(0, outfit_length - 1)
            random_idx = random.randint(0, self.num_items - 1)
            embeddings[replace_pos] = self.get_clip_feature(random_idx)
            label = [-0.9, -0.9] # 只要坏了，就不管完不完成，全部标负
        
        return {
            "embeddings": embeddings,
            "label": torch.tensor(label, dtype=torch.float)
        }

    @staticmethod
    def collate_fn(batch):
        embs = [item["embeddings"] for item in batch]
        labels = [item["label"] for item in batch]
        padded_embs = pad_sequence(embs, batch_first=True, padding_value=0.0)
        
        # 生成 mask
        batch_size = padded_embs.size(0)
        max_len = padded_embs.size(1)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        for i, outfit_embs in enumerate(embs):
            mask[i, :len(outfit_embs)] = False
            
        return {
            "item_embeddings": padded_embs,
            "mask": mask,
            "labels": torch.stack(labels) # [B, 2]
        }