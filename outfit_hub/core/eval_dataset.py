#outfit_hub/core/eval_dataset.py
import os
import json

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
