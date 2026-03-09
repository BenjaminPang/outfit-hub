import json
import os
import pandas as pd
from tqdm import tqdm
import pickle

from .base_processor import BaseProcessor
from ..utils.image_utils import get_image_md5
from ..tasks import FITBTaskEngine, CompatibilityTaskEngine
from ..utils.vector_db_utils import VectorDB


class Fashion32Processor(BaseProcessor):
    def process_category(self):
        category_list = ['all']
        self.category2idx = {category_str: i for i, category_str in enumerate(category_list)}
        self.idx2category = {i: category_str for i, category_str in enumerate(category_list)}
        self.category_len = len(category_list)

        save_path = os.path.join(self.output_path, 'category.json')
        with open(save_path, 'w') as f:
            json.dump(self.idx2category, f, ensure_ascii=False, indent=2)

    def parse_raw_data(self):
        outfits_data = []
        item_set = set()
        itemid2path = {}
        for split in ['Train', 'Valid', 'Test']:
            self.image_dir = os.path.join(self.root_path, f'Polyvore_Cate3_{split}')
            outfit_raw_ids = [x for x in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, x))]
            for outfit_raw_id in tqdm(outfit_raw_ids, desc=f"Processing {self.dataset_name}, {split} split."):
                try:
                    item_ids = []
                    item_filenames = os.listdir(os.path.join(self.image_dir, outfit_raw_id))
                    add_this_outfit = True
                    for filename in item_filenames:
                        item_image_path = os.path.join(self.image_dir, outfit_raw_id, filename)
                        if not os.path.exists(item_image_path):
                            add_this_outfit = False
                            break
                        item_id = get_image_md5(item_image_path)
                        itemid2path[item_id] = item_image_path
                        item_ids.append(item_id)

                    if len(item_filenames) > 2 and add_this_outfit:
                        outfits_data.append({
                            "item_ids": item_ids,
                            "split": split.lower()
                        })
                        item_set.update(item_ids)

                except Exception as e:
                    print(e)

        # Process item data
        for idx, item_id in enumerate(item_set):
            # process metadata
            category_id = 0
            category = "all"
            category_idx = 0
            item_entry = {
                'item_idx': idx,
                'item_id': item_id,
                'category_idx': category_idx,
                'category_id': category_id,
                'category': category,
                'ori_path': itemid2path[item_id],
                'source': self.dataset_name,
            }
            self.item_parquet.append(item_entry)
            self.itemid2itemidx[item_id] = idx

        # So far, it's confirmed that all the `outfit` values ​​in `outfits_data` are usable.
        # Process outfit data
        outfitid2outfit = {}
        for outfit_idx, outfit_data in enumerate(outfits_data):
            item_ids = outfit_data['item_ids']
            _, outfit_id = self.generate_outfit_id(item_ids)

            outfit_entry = {
                'outfit_id': outfit_id,
                'outfit_idx': outfit_idx,
                'item_ids': item_ids,
                'item_indices': [self.itemid2itemidx[item_id] for item_id in item_ids],
                'length': len(item_ids),
                'source': self.dataset_name,
                'split': outfit_data['split'],
            }
            outfitid2outfit[outfit_id] = outfit_entry
            self.outfit_parquet.append(outfit_entry)

        print(f"{self.dataset_name} main process finised.\nSummary: Number item: {len(self.item_parquet)}, Number outfit: {len(self.outfit_parquet)}, Number user: {len(self.user_parquet)}")

    def process_test(self):
        with open(os.path.join(self.output_path, 'clip_vision_features.pkl'), 'rb') as f:
            clip_feature = pickle.load(f)  # dict type
        vector_db = VectorDB(self.item_df, clip_feature, self.dataset_name)

        output_dir = os.path.join(self.output_path, "eval")

        fitb_task_engine = FITBTaskEngine()
        count_dict = {}
        for split in ['valid', 'test']:
            count = fitb_task_engine.generate(self.outfit_df, vector_db, output_dir, split=split, pool_size=100)
            count_dict[split] = count
        self.supported_tasks['fitb'] = count_dict

        compatibility_task_engine = CompatibilityTaskEngine()
        count_dict = {}
        for split in ['valid', 'test']:
            count = compatibility_task_engine.generate(self.outfit_df, vector_db, output_dir, split=split, pool_size=100)
            count_dict[split] = count
        self.supported_tasks['compatibility'] = count_dict
