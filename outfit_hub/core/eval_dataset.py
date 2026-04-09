#outfit_hub/core/eval_dataset.py
import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from .base_dataset import BaseOutfitDataset
from .datatypes import FashionItem, FashionOutfit, FashionComplementaryQuery, FashionCompatibilityData, FashionFillInTheBlankData


class FITBEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir, dataset_name, task_name, split='test', **kwargs):
        print(f"Loading FITB Data for evaluating...")
        super().__init__(root_dir, dataset_name, split=split, **kwargs)
        with open(os.path.join(self.dataset_dir, "eval", f"{task_name}_{split}.json"), 'r') as f:
            self.tasks = json.load(f)
        self._categories = self.items_df['category'].tolist()
        self._descriptions = self.items_df['description'].tolist()
        self._embedding_cache = self._load_vector_db_to_numpy()

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, task_idx):
        sample = self.tasks[task_idx]
        context_idxs = [idx for idx in sample['original_outfit'] if idx != sample['gt_item_idx']]
        candidate_idxs = sample['item_candidates']
        target_category = self._categories[int(sample['gt_item_idx'])]
        label = int(sample['gt_outfit_label'])

        incomplete_outfit, candidates = [], []
        for i, iidx in enumerate(context_idxs + candidate_idxs):
            item = self.construct_item(iidx)
            if i < len(context_idxs):
                incomplete_outfit.append(item)
            else:
                candidates.append(item)

        fitb_data = FashionFillInTheBlankData(
            query=FashionComplementaryQuery(
                outfit=incomplete_outfit,
                category=target_category
            ),
            label=label,
            candidates=candidates
        )
        return fitb_data

    @staticmethod
    def collate_fn(batch):
        query = [x['query'] for x in batch]
        label = [x['label'] for x in batch]
        candidates = [x['candidates'] for x in batch]
        return FashionFillInTheBlankData(
            query = query,
            label=label,
            candidates=candidates
        )


class CompEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir, dataset_name, task_name, split='test', **kwargs):
        print(f"Loading Compatibility Data for evaluating...")
        super().__init__(root_dir, dataset_name, split, **kwargs)
        with open(os.path.join(self.dataset_dir, "eval", f"{task_name}_{split}.json"), 'r') as f:
            self.tasks = json.load(f)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, task_idx):
        sample = self.tasks[task_idx]
        item_idxs = sample['items']
        label = sample['label']
        outfit = []
        for idx in item_idxs:
            entry = self.items_df.iloc[idx]
            image = self.get_image(idx, return_tensor=False)
            description = entry.get('description', '')
            embedding = self.get_feature(idx)
            item = FashionItem(
                item_id=idx,
                category=entry['category'],
                image=image,
                description=description,
                embedding=embedding,
                metadata=entry.to_dict()
            )
            outfit.append(item)
        compatibility_data = FashionCompatibilityData(
            query=FashionOutfit(
                outfit=outfit
            ),
            label=label
        )
        return compatibility_data


class AvailabilityMaskDataset(BaseOutfitDataset):
    def __init__(self, root_dir, dataset_name, seed=42, **kwargs):
        # 显式设置 split 为 'test'
        super().__init__(root_dir, dataset_name, split='test', **kwargs)
        self.missing_ratio = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        self.seed = seed
        
        self.item_pool = set()
        self._categories = self.items_df['category'].fillna('unknown').tolist()
        self._descriptions = self.items_df['description'].fillna('').tolist()
        self._embedding_cache = self._load_vector_db_to_numpy()

        self.samples = []
        for row in self.outfits_df.itertuples():
            item_idxs = row.item_indices
            if isinstance(item_idxs, str):
                item_idxs = json.loads(item_idxs)
            self.item_pool.update(item_idxs)
            self.samples.append(item_idxs)

        self.item_pool = sorted(list(self.item_pool))
        self.num_items = len(self.item_pool)
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        # 使用样本索引 i 结合全局 seed，确保每个 sample 的随机性是固定且可复现的
        sample_seed = self.seed + i
        rng = np.random.default_rng(sample_seed)
        
        full_outfit = self.samples[i]
        
        # 1. 随机挖掉一个作为 GT
        gt_idx_in_list = rng.integers(0, len(full_outfit))
        # gt_item_idx = full_outfit[gt_idx_in_list]
        # gt_item = self.construct_item(gt_item_idx)
        incomplete_idxs = [idx for j, idx in enumerate(full_outfit) if j != gt_idx_in_list]

        # 2. 构造模型输入
        item_list = [self.construct_item(idx) for idx in incomplete_idxs]
        
        # 生成所有 item 的随机置换
        shuffled_indices = rng.permutation(self.item_pool)

        output = {
            "query": FashionOutfit(
                outfit=item_list
            ),
            # "gt_item": gt_item,
            "candidate_indices": torch.from_numpy(shuffled_indices),
        }
        return output
    
    @staticmethod
    def collate_fn(batch):
        return {
            "query": [x['query'] for x in batch],
            # "gt_item": [x['gt_item'] for x in batch],
            "candidate_indices": torch.stack([x['candidate_indices'] for x in batch], dim=0)
        }
