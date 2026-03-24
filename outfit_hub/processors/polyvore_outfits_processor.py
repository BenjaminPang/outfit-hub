import json
import os
import csv
from tqdm import tqdm

from .base_processor import BaseProcessor


class PolyvoreOutfitsProcessor(BaseProcessor):
    include_description = True
    def __init__(self, dataset_name, dataset_config, img_size=224, chunk_size=50000):
        super().__init__(dataset_name, dataset_config, img_size, chunk_size)
        self.version = dataset_config['version']

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

    def _is_image_valid(self, item_id):
        image_path = os.path.join(self.image_dir, f"{item_id}.jpg")
        if os.path.exists(image_path):
            return True
        else:
            return False

    def parse_raw_data(self):
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

                if len(item_ids) > 1 and add_this_outfit:
                    outfits_data.append({
                        "item_ids": item_ids,
                        "split": split
                    })
                    item_set.update(item_ids)

        # Process item data
        for idx, item_id in enumerate(sorted(list(item_set))):
            # process metadata
            category_id = item_metadata[item_id]['category_id']
            category = item_metadata[item_id]["semantic_category"]
            category_idx = self.category2idx[category]
            description = item_metadata[item_id]["url_name"]
            item_entry = {
                'item_idx': idx,
                'item_id': item_id,
                'category_idx': category_idx,
                'category_id': category_id,
                'category': category,
                'description': description,
                'ori_path': os.path.join(self.image_dir, f"{item_id}.jpg"),
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

    def _transform_fitb_task(self, output_dir):
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
                original_outfit_context_id = entry['question'][0].split('_')[0]
                original_outfit = outfitrawid2outfit[original_outfit_context_id]
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
                gt_outfit_label = -1
                for i, n in enumerate(entry['answers']):
                    original_outfit_candidate_id, index = n.split('_')
                    # -1 because orignal file index start from 1 instead of 0
                    index = int(index) - 1
                    item_id = outfitrawid2outfit[original_outfit_candidate_id]['items'][index]['item_id']
                    candidate_idx = self.itemid2itemidx.get(item_id)

                    if original_outfit_context_id == original_outfit_candidate_id:
                        gt_outfit_label = i
                    if candidate_idx is not None:
                        candidates.append(candidate_idx)
                    else:
                        add_this_question = False
                        break
                    
                    outfit_candidate = item_idxs.copy()
                    outfit_candidate[target_pos] = candidate_idx
                    outfit_candidates.append(outfit_candidate)

                if add_this_question and gt_outfit_label != -1:
                    gt_item_idx = item_idxs[target_pos]
                    tasks.append({
                        "outfit_candidates": outfit_candidates, # Some methods can directly use this to evaluate
                        "gt_outfit_label": gt_outfit_label,  # GT is the first candidate
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

    def _transform_compatibility_task(self, output_dir):
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

    def save_tar(self):
        """
        Divide items into chunks and save them into multiple .tar files.
        This prevents creating excessively large single archives.
        After save all images to tar, remove the temp image folder.
        """
        tar_idx = 0
        for i in range(0, len(self.item_parquet), self.chunk_size):
            items_for_current_tar = self.item_parquet[i : i + self.chunk_size]
            tar_path = os.path.join(self.output_path, f"{tar_idx:03d}.tar")
            
            print(f"创建新 Tar 包: {tar_path} (包含 {len(items_for_current_tar)} 张)")
            self._parallel_save_to_tar(tar_path, items_for_current_tar, max_workers=8, target_size=self.img_size)
            
            if len(items_for_current_tar) == self.chunk_size:
                tar_idx += 1

    def process_test(self):
        output_dir = os.path.join(self.output_path, "eval")
        self._transform_fitb_task(output_dir)
        self._transform_compatibility_task(output_dir)