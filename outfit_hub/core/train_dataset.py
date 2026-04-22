import json
import random
import os

from tqdm import tqdm

from .base_dataset import BaseOutfitDataset
from .datatypes import FashionOutfit, FashionCompatibilityData, FashionContrastivetData, FashionComplementaryQuery


class FashionItemPoolDataset(BaseOutfitDataset):
    """
    返回所有训练套装中出现的唯一单品 ID 合集
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. 提取所有 items
        all_item_ids = []
        for items in self.outfits:
            all_item_ids.extend(items) # 将套装中的单品铺开
            
        # 2. 去重
        self.data = sorted(list(set(all_item_ids)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 覆写 getitem，确保返回的是单个单品对象
        item_id = self.data[index]
        return self.construct_item(item_id)


class NextItemPredictionDataset(BaseOutfitDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        if self.split == 'train':
            return len(self.outfits) * 5
        return len(self.outfits)

    def __getitem__(self, i: int) -> FashionContrastivetData:
        idx = i % len(self.outfits)
        item_idxs = self.outfits[idx]

        answer_idx_in_list = random.randint(0, len(item_idxs) - 1)
        answer_val = item_idxs[answer_idx_in_list]
        answer = self.construct_item(answer_val)

        remaining_idxs = [iidx for iidx in item_idxs if iidx != answer_val]
        # k means how many remaining items serve as outfit query
        k = random.randint(1, len(remaining_idxs))
        chosen_context_idxs = random.sample(remaining_idxs, k)
        item_list = [self.construct_item(iidx) for iidx in chosen_context_idxs]

        output = FashionContrastivetData(
            query=FashionComplementaryQuery(
                outfit=item_list,
                # category=answer.category
            ),
            answer=answer
        )

        return output

    @staticmethod
    def collate_fn(batch) -> FashionContrastivetData:
        query = [item['query'] for item in batch]
        answer = [item['answer'] for item in batch]
        
        return FashionContrastivetData(
            query=query,
            answer=answer
        )


class FashionCompatibilityPredictioneDataset(BaseOutfitDataset):
    """
    Value Function Dataset: Used to train the Value Head (VH).
    Purpose: By constructing positive, negative, and incomplete samples, it enables the model to learn how to evaluate the state of the current combination.
    """
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     if self.split == 'train':
    #         file_path = os.path.join(self.dataset_dir, 'anno', "compatibility_train.json")
    #     else:
    #         file_path = os.path.join(self.dataset_dir, "eval", f"compatibility_{self.split}.json")
    #     with open(file_path, 'r') as f:
    #         self.data = json.load(f)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.split == 'train':
            self.cat_to_indices = self.items_df.groupby('category').groups

            self.pos_data = [{"items": x, "label": 1} for x in self.outfits]
            self.neg_data = []
            for sample in tqdm(self.pos_data, desc="Generating mixed neg outfits"):
                items = sample['items']
                for _ in range(2):
                    self.neg_data.append({"items": self._generate_neg_v1(items), "label": 0})
                    self.neg_data.append({"items": self._generate_neg_v2(items), "label": 0})

            self.data = self.pos_data + self.neg_data
        else:
            file_path = os.path.join(self.dataset_dir, "eval", f"compatibility_{self.split}.json")
            with open(file_path, 'r') as f:
                self.data = json.load(f)

    def _generate_neg_v1(self, item_idxs: list[int]) -> list[int]:
        """
        把所有的单品都变成同类别的任意一件单品组成negative outfit
        """
        neg_outfit = []
        for item_idx in item_idxs:
            target_cat = self._categories[item_idx]
            pool = self.cat_to_indices[target_cat]
            
            if len(pool) > 1:
                # 排除掉当前那件，选个不一样的
                choices = [i for i in pool if i != item_idx]
                neg_outfit.append(random.choice(choices))
            else:
                # 如果该类实在没别的了，全库随机抽一件
                neg_outfit.append(random.randint(0, len(self.items_df) - 1))
            
        return neg_outfit
    
    def _generate_neg_v2(self, item_idxs: list[int]) -> list[int]:
        """
        视觉冗余负样本：挑选一个单品，寻找与其最相似的 n-1 个物品组成装扮。
        这类负样本通常会导致“满屏全是同类单品”的效果。
        """
        if not item_idxs:
            return []
        
        # 1. 随机从原装扮里选一个单品作为“种子” (Anchor)
        anchor_idx = random.choice(item_idxs)
        target_len = len(item_idxs) # 保持原装扮长度
        
        # 2. 从向量数据库中寻找最接近的邻居
        # 我们需要找 target_len - 1 个邻居
        try:
            neighbors = self.vector_db.get_nearest_neighbors_ids(anchor_idx, target_len - 1)
        except Exception as e:
            print(f"VectorDB search failed: {e}")
            neighbors = []

        # 3. 组合成负装扮
        neg_outfit = [anchor_idx] + neighbors
        
        # 降级处理：如果 DB 没搜够（虽然概率很低），用随机补齐
        while len(neg_outfit) < target_len:
            neg_outfit.append(random.randint(0, len(self.items_df) - 1))
            
        return neg_outfit

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