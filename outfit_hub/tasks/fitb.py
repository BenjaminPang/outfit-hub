# outfit_hub/tasks/fitb.py

import os
import random
import json

import pandas as pd
from tqdm import tqdm


class FITBTaskEngine:
    """
    Task: Fill-In-The-Blank (FITB)
    Metric: Accuracy
    Description: Given an outfit with one missing item, select the correct item from N candidates.
    """
    
    @staticmethod
    def generate(outfit_df, vector_db, output_dir, split="train", pool_size=100, num_candidates=4):
        """
        Builds the FITB test set with hard negatives from a vector database.
        Args:
            pool_size: If -1, use random sampling. 
                       If > 0, use Hard Negative search via vector_db.

        """
        target_df = outfit_df[outfit_df['split'] == split]
        tasks = []
        for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Generating FITB Tasks"):
            item_idxs = list(row['item_indices']) 
            
            # 1. Select a target position to 'blank'
            target_pos = random.choice(range(len(item_idxs)))
            gt_item_idx = item_idxs[target_pos]
            
            # 2. Find Hard Negatives using vector_db
            gt_emb = vector_db.get_embeddings(gt_item_idx)
            # Fetch more than needed to filter out the ground truth
            if pool_size > 0:
                search_results = vector_db.search(gt_emb, k=pool_size + 1)
                
                neg_pool = [res[0] for res in search_results if int(res[0]) != int(gt_item_idx)]
                
                # Fallback if pool is too small
                if len(neg_pool) < (num_candidates - 1):
                    all_ids = vector_db.item_df['item_idx'].tolist()
                    neg_pool = random.sample(all_ids, num_candidates - 1)
                
                selected_negs = random.sample(neg_pool[:pool_size], num_candidates - 1)
            elif pool_size == -1:
                neg_pool = vector_db.item_df['item_idx'].tolist()
                neg_pool.remove(gt_item_idx)
                selected_negs = random.sample(neg_pool, num_candidates - 1)
            
            # 3. Construct Candidates (GT is always at index 0 for consistency, shuffle later in DataLoader)
            candidates = [int(gt_item_idx)] + [int(idx) for idx in selected_negs]
            outfit_candidates = []
            for c_idx in candidates:
                outfit_candidate = item_idxs.copy()
                outfit_candidate[target_pos] = c_idx
                outfit_candidates.append(outfit_candidate)
            
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
        print(f"FITB Task saved to {output_path}")

        return len(tasks)

    @staticmethod
    def transform(file_path):
        """
        Transform raw fitb file to standard format.
        """
        pass

    @staticmethod
    def load(file_path):
        """
        Loads the FITB task file.
        """
        pass