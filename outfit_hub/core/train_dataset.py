import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader

from .base_dataset import BaseOutfitDataset
    

class OutfitTrainDataset(BaseOutfitDataset):
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


def get_combined_loader(dataset_names, root_dir="./data", split='train', batch_size=32, num_workers=4, max_seq_length=9, pin_memory=True, transform=None, load_clip=True, load_img=False):
    """
    dataset_configs: List[str], such as:
    ["polyvoreu519", "ifashion"]
    """
    datasets = []
    for ds_idx, name in enumerate(dataset_names):
        ds = OutfitTrainDataset(
            root_dir=root_dir, 
            dataset_name=name, 
            dataset_idx=ds_idx,
            split=split,
            max_seq_length=max_seq_length,
            transform=transform,
            load_clip=load_clip,
            load_img=load_img,
        )
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)
    
    return DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=(split == "train"),
        collate_fn=OutfitTrainDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )