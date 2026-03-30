import json
import random
import os
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from .base_dataset import BaseOutfitDataset
# from .datatypes import , 
from external.outfit_transformer.src.data.datatypes import FashionCompatibilityQuery, FashionItem, FashionCompatibilityData
    

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


class FashionCompatibilityPredictioneDataset(BaseOutfitDataset):
    """
    Value Function Dataset: Used to train the Value Head (VH).
    Purpose: By constructing positive, negative, and incomplete samples, it enables the model to learn how to evaluate the state of the current combination.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split == 'train':
            file_path = os.path.join(self.dataset_dir, 'anno', "compatibility_train.json")
        else:
            file_path = os.path.join(self.dataset_dir, "eval", f"compatibility_{self.split}.json")
        with open(file_path, 'r') as f:
            self.data = json.load(f)

        self._categories = self.items_df['category'].tolist()
        self._descriptions = self.items_df['description'].tolist()
        self._embedding_cache = self._load_vector_db_to_numpy()

    def __getitem__(self, i: int) -> FashionCompatibilityData:
        sample = self.data[i]
        outfit = FashionCompatibilityQuery(
            outfit=[self.construct_item(iidx) for iidx in sample['items']]
        )
        output = FashionCompatibilityData(
            label=sample['label'],
            query=outfit
        )
        return output
    
    def __len__(self):
        return len(self.data)

    def construct_item(self, iidx: int) -> FashionItem:
        try:
            return FashionItem(
                category= self._categories[iidx],
                image=None,
                description= self._descriptions[iidx],
                embedding=self.get_feature(iidx)
            )
        except IndexError:
            return None

    @staticmethod
    def collate_fn(batch):
        label = [item['label'] for item in batch]
        query = [item['query'] for item in batch]
        
        return FashionCompatibilityData(
            label=label,
            query=query
        )