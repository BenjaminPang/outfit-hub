import json
import random
import os

import torch
from torch.nn.utils.rnn import pad_sequence

from .base_dataset import BaseOutfitDataset
from .datatypes import FashionOutfit, FashionCompatibilityData, FashionTripletData, FashionComplementaryQuery
    

class NextItemPredictionDataset(BaseOutfitDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = [json.loads(x) for x in self.outfits_df['item_indices'].tolist()]

        self._categories = self.items_df['category'].tolist()
        self._descriptions = self.items_df['description'].tolist()
        self._embedding_cache = self._load_vector_db_to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> FashionTripletData:
        item_idxs = self.data[i]
        answer_index = random.randint(0, len(item_idxs) - 1)
        item_list = [self.construct_item(iidx) for iidx in item_idxs if iidx != answer_index]
        answer = self.construct_item(item_idxs[answer_index])
        output = FashionTripletData(
            query=FashionComplementaryQuery(
                outfit=item_list,
                category=answer.category
            ),
            answer=answer
        )

        return output

    @staticmethod
    def collate_fn(batch) -> FashionTripletData:
        query = [item['query'] for item in batch]
        answer = [item['answer'] for item in batch]
        
        return FashionTripletData(
            query=query,
            answer=answer
        )


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
        outfit = FashionOutfit(
            outfit=[self.construct_item(iidx) for iidx in sample['items']]
        )
        output = FashionCompatibilityData(
            label=sample['label'],
            query=outfit
        )
        return output
    
    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        label = [item['label'] for item in batch]
        query = [item['query'] for item in batch]
        
        return FashionCompatibilityData(
            label=label,
            query=query
        )