import json
import os
import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor


class PolvyoreUProcessor(BaseProcessor):
    include_description = False
    def __init__(self, dataset_name, dataset_config, img_size=224, chunk_size=50000):
        super().__init__(dataset_name, dataset_config, img_size, chunk_size)
        self.version = dataset_config['version']
        if self.version == "519":
            self.pos_config = [(1, 'top'), (2, 'top'), (3, 'bottom'), (4, 'shoe')]
        elif self.version == "630":
            self.pos_config = [(1, 'top'), (2, 'bottom'), (3, 'shoe')]
        else:
            raise ValueError(f"Version {self.version} not found.")
        self.outfit_len = len(self.pos_config)

    def process_category(self):
        category_list = ['top', 'bottom', 'shoe']
        self.category2idx = {category_str: i for i, category_str in enumerate(category_list)}
        self.idx2category = {i: category_str for i, category_str in enumerate(category_list)}
        self.category_len = len(category_list)

        save_path = os.path.join(self.output_path, 'category.json')
        with open(save_path, 'w') as f:
            json.dump(self.idx2category, f, indent=2)

    def _is_image_valid(self, item_id):
        img_path = os.path.join(self.image_dir, f"{item_id}.jpg")
        if not os.path.exists(img_path):
            return False
        
        return True
    
    def _get_item_mapping_dict(self):
        itemid2category = {}
        item_mapping_dict = {}
        for category in self.category2idx.keys():
            with open(os.path.join(self.root_path, f'image_list_{category}'), 'r') as f:
                items = [os.path.splitext(line.strip())[0] for line in f]
                item_mapping_dict[category] = items
                # 批量更新映射
                for iid in items:
                    itemid2category[iid] = category
        return itemid2category, item_mapping_dict

    def parse_raw_data(self):
        itemid2category, item_mapping_dict = self._get_item_mapping_dict()

        item_set = set()
        outfits_data = []
        
        outfit_num_mapping = {'train': 83416, 'valid': 8736, 'test': 14654}
        # load outfit raw data
        for ori_split in ['train', 'val', 'test']:
            split = "valid" if ori_split == 'val' else ori_split
            file_path = os.path.join(self.root_path, f'tuples_{ori_split}_posi')
            
            with open(file_path, 'r') as f:
                # 跳过第一行
                next(f) 
                
                for line in tqdm(f, total=outfit_num_mapping[split], desc=f"Processing {split} outfit data and building outfits_data list and item_set"):
                    parts = line.strip().split(',')
                    user_id = parts[0]
                    item_ids = [
                        item_mapping_dict[cat][int(parts[idx])]
                        for idx, cat in self.pos_config
                        if int(parts[idx]) != -1
                    ]

                    add_this_outfit = True
                    for item_id in item_ids:
                        if not self._is_image_valid(item_id):
                            add_this_outfit = False
                            break

                    if len(item_ids) > 2 and add_this_outfit and user_id:
                        outfits_data.append({
                            "user_id": user_id,
                            "item_ids": item_ids,
                            "split": split
                        })
                        item_set.update(item_ids)

        # Process item data
        for idx, item_id in enumerate(item_set):
            # process metadata
            category = itemid2category[item_id]
            category_id = self.category2idx[category]
            category_idx = category_id
            item_entry = {
                'item_idx': idx,
                'item_id': item_id,
                'category_idx': category_idx,
                'category_id': category_id,
                'category': category,
                'ori_path': os.path.join(self.image_dir, f"{item_id}.jpg"),
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
                'split': outfit_data['split'],
            }
            outfitid2outfit[outfit_id] = outfit_entry
            self.outfit_parquet.append(outfit_entry)

            user_id = outfit_data['user_id']
            if user_id not in user_outfit_dict.keys():
                user_outfit_dict[user_id] = [outfit_idx]
            else:
                user_outfit_dict[user_id].append(outfit_idx)

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

    def _transform_fitb_task(self, output_dir, item_mapping_dict):
        # First transform fitb
        tasks = []
        with open(os.path.join(self.root_path, f"fill_in_blank_test"), 'r') as f:
            next(f) 
            
            for line in tqdm(f, total=14655, desc=f"Transform fitb test"):
                add_this_question = True
                parts = line.strip().split(',')
                outfit_candidates = []
                
                # There are 4 groups in total. 519 is a group of 5 elements, and 630 is a group of 4 elements.
                for i in range(0, len(parts), self.outfit_len + 1):
                    outfit_chunk = parts[i : i + self.outfit_len + 1]
                    
                    # Parse the 5 fields in this group
                    user_id = outfit_chunk[0]
                    item_ids = [
                        item_mapping_dict[cat][int(outfit_chunk[idx])]
                        for idx, cat in self.pos_config
                        if int(outfit_chunk[idx]) != -1
                    ]
                    
                    item_idxs = []
                    for item_id in item_ids:
                        item_idx = self.itemid2itemidx.get(item_id)
                        if item_idx is not None:
                            item_idxs.append(item_idx)
                        else:
                            add_this_question = False
                            break
                    outfit_candidates.append(item_idxs)

                if not add_this_question:
                    continue
                
                target_pos = -1
                candidates = []
                for target_pos in range(len(outfit_candidates[0])):
                    if outfit_candidates[0][target_pos] != outfit_candidates[1][target_pos]:
                        candidates = [x[target_pos] for x in outfit_candidates]
                        break
                if target_pos == -1:
                    continue

                if add_this_question:
                    tasks.append({
                        "outfit_candidates": outfit_candidates, # Some methods can directly use this to evaluate
                        "gt_outfit_label": 0,  # GT is the first candidate
                        "original_outfit": outfit_candidates[0],
                        "blank_position": target_pos,
                        "gt_item_idx": outfit_candidates[0][target_pos],
                        "item_candidates": candidates,
                        "user_idx": self.userid2useridx[user_id]
                    })
                
            # Save to json
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"fitb_test.json")
            with open(output_path, 'w') as f:
                json.dump(tasks, f, indent=2)
            print(f"FITB Task saved to {output_path}, total {len(tasks)} question")
        self.supported_tasks['fitb'] = {'test': len(tasks)}

    def _transform_compatibility_task(self, output_dir, item_mapping_dict, is_hard=False):
        count_dict = {}
        for split in ['val', 'test']:
            tasks = []
            raw_data_path_list = [os.path.join(self.root_path, f'tuples_{split}_{x}') for x in ['nega', "posi"]]
            if is_hard:
                raw_data_path_list[0] = os.path.join(self.root_path, f'tuples_{split}_nega_hard')
            for label, raw_data_path in enumerate(raw_data_path_list):
                with open(raw_data_path, 'r') as f:
                    next(f)
                    for line in f:
                        parts = line.strip().split(',')
                        user_id = parts[0]
                        item_ids = [
                            item_mapping_dict[cat][int(parts[idx])]
                            for idx, cat in self.pos_config
                            if int(parts[idx]) != -1
                        ]

                        item_idxs = []
                        add_this_question = True
                        for item_id in item_ids:
                            item_idx = self.itemid2itemidx.get(item_id)
                            if item_idx is not None:
                                item_idxs.append(item_idx)
                            else:
                                add_this_question = False
                                break
                            
                        if add_this_question:
                            tasks.append({
                                "items": [int(idx) for idx in item_idxs],
                                "label": label,
                                "user_idx": self.userid2useridx[user_id]
                            })

            os.makedirs(output_dir, exist_ok=True)
            # Use a descriptive name including pool_size to distinguish strategies
            if is_hard:
                output_path = os.path.join(output_dir, f"compatibility_hard_{split}.json")
            else:
                output_path = os.path.join(output_dir, f"compatibility_{split}.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2)
                
            print(f"✅ Compatibility Task saved to {output_path}, Number {len(tasks)}")

            count_dict[split] = len(tasks)

        if is_hard:
            self.supported_tasks['compatibility_hard'] = count_dict
        else:
            self.supported_tasks['compatibility'] = count_dict

    def process_test(self):
        output_dir = os.path.join(self.output_path, "eval")
        itemid2category, item_mapping_dict = self._get_item_mapping_dict()
        self._transform_fitb_task(output_dir, item_mapping_dict)
        self._transform_compatibility_task(output_dir, item_mapping_dict, is_hard=False)
        self._transform_compatibility_task(output_dir, item_mapping_dict, is_hard=True)
