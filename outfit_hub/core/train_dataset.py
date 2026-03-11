import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader

from .base_dataset import BaseOutfitDataset
    

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


def get_combined_loader(dataset_configs, split='train', batch_size=32, **kwargs):
    """
    dataset_configs: List[dict], such as:
    [
        {"name": "polyvoreu519", "load_img": True},
        {"name": "ifashion", "load_img": True}
    ]
    """
    datasets = []
    
    for config in dataset_configs:
        ds = OutfitTrainDataset(
            root_dir="data", 
            dataset_name=config['name'], 
            split=split,
            load_img=config.get('load_img', False),
            **kwargs
        )
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)
    
    return DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'),
        collate_fn=OutfitTrainDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )