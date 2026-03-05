import json
import os
import csv
import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor
from ..tasks import FITBTaskEngine, CompatibilityTaskEngine


class PolyvoreOutfitsProcessor(BaseProcessor):
    def __init__(self, dataset_name, dataset_config, img_size=224, chunk_size=50000):
        super().__init__(dataset_name, dataset_config, img_size, chunk_size)
        self.version = dataset_config['version']
        self.temp_image_save_path = os.path.join(self.output_path, "temp_images")
        os.makedirs(self.temp_image_save_path, exist_ok=True)

    def process_category(self):
        category_file_path = os.path.join(self.root_path, "categories.csv")
        category_set = set()
        with open(category_file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # row 是一个列表，例如 ['11', 'sweater', 'tops']
                _, category1, category2 = row
                category_set.add(category1)
                category_set.add(category2)

        category_list = list(category_set)
        category_list.remove("")
        self.category2idx = {category_str: i for i, category_str in enumerate(category_list)}
        self.idx2category = {i: category_str for i, category_str in enumerate(category_list)}
        self.category_len = len(category_list)

        save_path = os.path.join(self.output_path, 'category.json')
        with open(save_path, 'w') as f:
            json.dump(self.idx2category, f, indent=2)

    def _save_image_from_parquet(self, item_id, image_byte):
        """
        将 parquet 中的字节流保存为临时图片文件
        """       
        # 3. 构造完整的文件路径（根据图片内容，通常是 .jpg）
        full_file_path = os.path.join(self.temp_image_save_path, f"{item_id}.jpg")
        if os.path.exists(full_file_path):
            return True
    
        try:
            with open(full_file_path, "wb") as f:
                f.write(image_byte)
            return True
        except Exception as e:
            print(f"Error saving image {item_id}: {e}")
            return False

    def _is_image_valid(self, item_id):
        result = self.image_data[self.image_data['item_id'] == item_id]
        
        if result.empty:  # Pandas 推荐使用 .empty 检查是否为空
            return False
        
        try:
            image_byte = result.iloc[0]["image.bytes"]
            return self._save_image_from_parquet(item_id, image_byte)
            
        except Exception as e:
            # 捕获可能出现的其他错误（如字段缺失或 IO 错误）
            print(f"Error processing item {item_id}: {e}")
            return False

    def parse_raw_data(self):
        # Load image parquet
        self.image_data = pd.concat([pd.read_parquet(os.path.join(self.image_dir, f"{fn}.parquet")) for fn in ["train", "validation", "test"]], ignore_index=True)

        with open(os.path.join(self.root_path, 'polyvore_item_metadata.json'), 'r') as f:
            item_metadata = json.load(f)

        item_set = set()
        outfits_data = []
        for split in ['train', 'valid', 'test']:
            with open(os.path.join(self.root_path, self.version, f'{split}.json'), 'r') as f:
                raw_outfits_data = json.load(f)

            for raw_outfit in tqdm(raw_outfits_data, desc=f"Processing {split} outfits"):
                item_ids = [x['item_id'] for x in raw_outfit['items']]

                add_this_outfit = True
                for item_id in item_ids:
                    if not self._is_image_valid(item_id):
                        add_this_outfit = False
                        break

                if len(item_ids) > 2 and add_this_outfit:
                    outfits_data.append({
                        "item_ids": item_ids,
                        "split": split
                    })
                    item_set.update(item_ids)

        # Process item data
        for idx, item_id in enumerate(item_set):
            # process metadata
            category_id = item_metadata[item_id]['category_id']
            category = item_metadata[item_id]["semantic_category"]
            category_idx = self.category2idx[category]
            item_entry = {
                'item_idx': idx,
                'item_id': item_id,
                'category_idx': category_idx,
                'category_id': category_id,
                'category': category,
                'ori_path': os.path.join(self.temp_image_save_path, f"{item_id}.jpg"),
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

        # No user is involved in this dataset
        self.user_parquet = []

        print(f"{self.dataset_name} main process finised.\nSummary: Number item: {len(self.item_parquet)}, Number outfit: {len(self.outfit_parquet)}, Number user: {len(self.user_parquet)}")
        
    def process_test(self):
        output_dir = os.path.join(self.output_path, "eval")

        # First transform fitb
        count_dict = {}
        for split in ['valid', 'test']:
            with open(os.path.join(self.root_path, self.version, f'{split}.json'), 'r') as f:
                raw_outfits_data = json.load(f)
            outfitrawid2outfit = {outfit['set_id']: outfit for outfit in raw_outfits_data}

            with open(os.path.join(self.root_path, self.version, f"fill_in_blank_{split}.json"), 'r') as f:
                fitb_raw_data = json.load(f)
            
            tasks = []
            for entry in tqdm(fitb_raw_data, desc=f"Generating FITB Tasks for {split}"):
                add_this_question = True
                target_pos = int(entry['blank_position']) - 1
                original_outfit_raw_id = entry['question'][0].split('_')[0]
                original_outfit = outfitrawid2outfit[original_outfit_raw_id]
                item_idxs = []
                for x in original_outfit['items']:
                    item_idx = self.itemid2itemidx.get(x['item_id'])
                    if item_idx is not None:
                        item_idxs.append(item_idx)
                    else:
                        add_this_question = False
                        break

                if not add_this_question:
                    continue

                candidates = []
                outfit_candidates = []
                for n in entry['answers']:
                    outfit_id, index = n.split('_')
                    # -1 because orignal file index start from 1 instead of 0
                    index = int(index) - 1
                    item_id = outfitrawid2outfit[outfit_id]['items'][index]['item_id']
                    candidate_idx = self.itemid2itemidx.get(item_id)
                    if candidate_idx is not None:
                        candidates.append(candidate_idx)
                    else:
                        add_this_question = False
                        break
                    
                    outfit_candidate = item_idxs.copy()
                    outfit_candidate[target_pos] = candidate_idx
                    outfit_candidates.append(outfit_candidate)

                if add_this_question:
                    gt_item_idx = item_idxs[target_pos]
                    tasks.append({
                        "outfit_candidates": outfit_candidates, # Some methods can directly use this to evaluate
                        "gt_outfit_label": 0,  # GT is the first candidate
                        "original_outfit": item_idxs,
                        "blank_position": target_pos,
                        "gt_item_idx": gt_item_idx,
                        "item_candidates": candidates,
                    })
                
            # Save to json
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"fitb_{split}.json")
            with open(output_path, 'w') as f:
                json.dump(tasks, f, indent=2)
            print(f"FITB Task saved to {output_path}, total {len(tasks)} question")
            count_dict[split] = len(tasks)
        self.supported_tasks['fitb'] = count_dict

        # Then process compatibility task
        count_dict = {}
        for split in ['valid', 'test']:
            with open(os.path.join(self.root_path, self.version, f'{split}.json'), 'r') as f:
                raw_outfits_data = json.load(f)
            outfitrawid2outfit = {outfit['set_id']: outfit for outfit in raw_outfits_data}
            with open(os.path.join(self.root_path, self.version, f"compatibility_{split}.txt"), 'r') as f:
                tasks = []
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    label = int(parts[0])
                    item_idxs = []
                    add_this_question = True
                    for raw_item in parts[1:]:
                        original_outfit_raw_id, index = raw_item.split('_')
                        index = int(index) - 1
                        
                        original_outfit = outfitrawid2outfit[original_outfit_raw_id]
                        item_id = original_outfit['items'][index]['item_id']
                        item_idx = self.itemid2itemidx.get(item_id)
                        if item_idx is not None:
                            item_idxs.append(item_idx)
                        else:
                            add_this_question = False
                        
                    if add_this_question:
                        tasks.append({
                            "items": [int(idx) for idx in item_idxs],
                            "label": label
                        })

            os.makedirs(output_dir, exist_ok=True)
            # Use a descriptive name including pool_size to distinguish strategies
            output_path = os.path.join(output_dir, f"compatibility_{split}.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2)
                
            print(f"✅ Compatibility Task saved to {output_path}, Number {len(tasks)}")

            count_dict[split] = len(tasks)

        self.supported_tasks['compatibility'] = count_dict
