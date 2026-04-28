#outfit_hub/core/eval_dataset.py
import os
import json
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from .base_dataset import BaseOutfitDataset
from .datatypes import FashionItem, FashionOutfit, FashionComplementaryQuery, FashionCompatibilityData, FashionFillInTheBlankData


class FITBEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir: str, dataset_name: str, feature_path: str,  split: str = 'test',**kwargs):
        super().__init__(root_dir, dataset_name, feature_path,  split=split, **kwargs)
        print(f"Loading FITB Data for evaluating...")
        split = kwargs.get("split", 'test')
        with open(os.path.join(self.dataset_dir, "eval", f"fitb_{split}.json"), 'r') as f:
            self.tasks = json.load(f)

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
    
    def get_retrieval_gallery_map(self, threshold=2999):
        """
        根据论文要求：
        1. 筛选出样本数 >= 3000 的品类
        2. 为每个品类预随机选出 2999 个干扰项索引
        """
        my_random = random.Random(0)
        cat_to_indices = defaultdict(list)
        for idx, cat in enumerate(self._categories):
            cat_to_indices[cat].append(idx)
        
        gallery_map = {}
        for cat, indices in cat_to_indices.items():
            if len(indices) >= threshold:
                gallery_map[cat] = my_random.sample(indices, threshold)
        return gallery_map


class CompEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir: str, dataset_name: str, feature_path: str,  split: str = 'test',**kwargs):
        print(f"Loading Compatibility Data for evaluating...")
        super().__init__(root_dir, dataset_name, feature_path,  split=split, **kwargs)
        split = kwargs.get("split", 'test')
        with open(os.path.join(self.dataset_dir, "eval", f"compatibility_{split}.json"), 'r') as f:
            self.tasks = json.load(f)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, task_idx):
        sample = self.tasks[task_idx]
        item_idxs = sample['items']
        label = sample['label']
        return {
            "outfit": FashionOutfit(
                outfit=[self.construct_item(idx) for  idx in item_idxs]
            ),
            "label": label
        }
    
    @staticmethod
    def collate_fn(batch):
        return {
            "query": [x["outfit"] for x in batch],
            "label": torch.tensor([x['label'] for x in batch], dtype=int)
        }


class OutfitGenerationEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir: str, dataset_name: str, feature_path: str,  split: str = 'test',**kwargs):
        super().__init__(root_dir, dataset_name, feature_path,  split=split, **kwargs)
        self.seed = 0
        random.seed(self.seed)

        all_item_idxs = set()
        for item_idxs in self.outfits:
            all_item_idxs.update(item_idxs)
        self.item_pool = list(sorted(all_item_idxs))

        category_map = {
            "tops": [],
            "bottoms": [],
            "all-body": [],
            "outerwear": [],
            "shoes": [],
            "accessories": [],
        }

        for iidx in all_item_idxs:
            cat = str(self._categories[iidx])
            if cat in category_map.keys():
                category_map[cat].append(iidx)

        self.samples = []
        for cat, pool in category_map.items():
            random.shuffle(pool)
            # 安全采样：取 count 和 实际长度 的最小值
            self.samples.extend(pool[:1000])

        # x = 2
        # self.samples = self.samples[0:x] + self.samples[1000:1000+x] + self.samples[2000:2000+x] + self.samples[3000:3000+x] + self.samples[4000:4000+x] + self.samples[5000:5000+x]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        # 使用样本索引 i 结合全局 seed，确保每个 sample 的随机性是固定且可复现的
        idx = self.samples[i]
        initial_outfit = [self.construct_item(idx)]

        output = {
            "start_outfit": FashionOutfit(
                outfit=initial_outfit,
            )
        }
        return output
    
    @staticmethod
    def collate_fn(batch):
        return {
            "start_outfit": [x['start_outfit'] for x in batch]
        }


class DistortionRatioEvalDataset(BaseOutfitDataset):
    def __init__(self, root_dir, dataset_name, seed=42, **kwargs):
        super().__init__(root_dir, dataset_name, split='test', **kwargs)
        self.missing_ratio = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        self.seed = seed
        
        self.item_pool = set()

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
        
        # 1. 随机挖掉一个作为 start item
        start_item_idx = full_outfit[rng.integers(0, len(full_outfit))]
        start_item = self.construct_item(start_item_idx)

        # 2. 构造模型输入
        # item_list = [self.construct_item(idx) for idx in full_outfit]
        
        # 生成所有 item 的随机置换
        shuffled_indices = rng.permutation(self.item_pool)

        output = {
            "length": len(full_outfit),
            "start_outfit": FashionOutfit(
                outfit=[start_item]
            ),
            "candidate_indices": torch.from_numpy(shuffled_indices),
        }
        return output
    
    @staticmethod
    def collate_fn(batch):
        return {
            "length": [x['length'] for x in batch],
            "start_outfit": [x['start_outfit'] for x in batch],
            "candidate_indices": torch.stack([x['candidate_indices'] for x in batch], dim=0)
        }
