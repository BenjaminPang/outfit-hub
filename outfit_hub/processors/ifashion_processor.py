# outfit_hub/precessors/ifashion_processor.py
import json
import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

from .base_processor import BaseProcessor
from ..tasks import FITBTaskEngine, CompatibilityTaskEngine
from ..utils.vector_db_utils import VectorDB


class iFashionProcessor(BaseProcessor):
    def process_category(self):
        category_file_path = os.path.join(self.root_path, "cate_id2text.json")
        self.raw_category_id2string = {}
        with open(category_file_path, 'r') as f:
            self.raw_category_id2string = json.load(f)
        category_list = list(self.raw_category_id2string.values())
        self.category2idx = {category_str: i for i, category_str in enumerate(category_list)}
        self.idx2category = {i: category_str for i, category_str in enumerate(category_list)}
        self.category_len = len(category_list)

        save_path = os.path.join(self.output_path, 'category.json')
        with open(save_path, 'w') as f:
            json.dump(self.idx2category, f, indent=2)

    def _is_image_valid(self, item_id):
        category_id = self.itemid2item[item_id].get("cate_id", "")  # this id is a long hex id like 'a6cd655d1645f38869be793d5fb194cb'. Need to be transformed into string
        category = self.raw_category_id2string.get(category_id, "")
        if not category:
            return False
        img_path = os.path.join(self.image_dir, category, f"{item_id}.png")
        if not os.path.exists(img_path):
            return False
        
        #TODO More strictly, we can open the image to further validate the image file. Omit for now.
        
        return True

    def parse_raw_data(self):
        # load item raw data
        with open(f'{self.root_path}/item_data.json', 'r') as f:
            item_json_data = json.load(f)
        self.itemid2item = {raw_item['item_id']: raw_item for raw_item in item_json_data}

        outfitid2userid = {}
        with open(f'{self.root_path}/user_data.txt', 'r') as f:
            for line in tqdm(f, total=19191117, desc="Loading user data and building outfitid2userid dict"):
                line = line.strip()
                if not line:
                    continue
                user_id, item_ids_str, raw_outfit_id = line.split(',')
                outfitid2userid[raw_outfit_id] = user_id

        # load outfit raw data
        outfits_data = []
        item_set = set()
        with open(f"{self.root_path}/outfit_data.txt", 'r') as f:
            for line in tqdm(f, total=1013136, desc="Processing outfit data and building outfits_data list and item_set"):
                line = line.strip()
                if not line:
                    continue
                raw_outfit_id, temp = line.split(',')
                item_ids = temp.split(';')
                add_this_outfit = True
                for item_id in item_ids:
                    if not self._is_image_valid(item_id):
                        add_this_outfit = False
                        break

                user_id = outfitid2userid.get(raw_outfit_id, "")
                if raw_outfit_id and len(item_ids) > 2 and add_this_outfit and user_id:
                    outfits_data.append({
                        "user_id": user_id,
                        "item_ids": item_ids
                    })
                    item_set.update(item_ids)

        # Process item data
        for idx, item_id in enumerate(item_set):
            # process metadata
            category_id = self.itemid2item[item_id]['cate_id']
            category = self.raw_category_id2string.get(category_id, "")
            category_idx = self.category2idx[category]
            item_entry = {
                'item_idx': idx,
                'item_id': item_id,
                'category_idx': category_idx,
                'category_id': category_id,
                'category': category,
                'ori_path': os.path.join(self.image_dir, category, f"{item_id}.png"),
                'source': self.dataset_name,
            }
            self.item_parquet.append(item_entry)
            self.itemid2itemidx[item_id] = idx

        # So far, it's confirmed that all the `outfit` values ​​in `outfits_data` are usable.
        # Process outfit data
        user_outfit_dict = {}
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
                'split': 'train',
            }
            outfitid2outfit[outfit_id] = outfit_entry
            self.outfit_parquet.append(outfit_entry)

            user_id = outfit_data['user_id']
            if user_id not in user_outfit_dict.keys():
                user_outfit_dict[user_id] = [outfit_idx]
            else:
                user_outfit_dict[user_id].append(outfit_idx)

        # We also need to extract test and valid outfit from raw data
        all_item_image_path = np.load(f'{self.root_path}/all_item_image_paths.npy',allow_pickle=True)
        valid_outfit_raw = np.load(f'{self.root_path}/valid_grd.npy',allow_pickle=True).item()
        test_outfit_raw = np.load(f'{self.root_path}/test_grd.npy',allow_pickle=True).item()

        for outfits_raw, split in zip([valid_outfit_raw, test_outfit_raw], ['valid', 'test']):
            for outfits in outfits_raw.values():
                item_ids = [all_item_image_path[x].split('/')[-1].split('.')[0] for x in outfits['outfits']]
                _, outfit_id = self.generate_outfit_id(item_ids)
                outfit_entry = outfitid2outfit.get(outfit_id, {})
                if outfit_entry:
                    outfit_entry['split'] = split

        # process user parquet
        for user_idx, (user_id, outfit_indices) in enumerate(user_outfit_dict.items()):
            self.user_parquet.append({
                'user_id': user_id,
                'user_idx': user_idx,
                'outfit_indices': outfit_indices,
                'outfit_num': len(outfit_indices),
                'source': self.dataset_name,
            })

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
