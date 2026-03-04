import os
import random
import json
import pandas as pd
from tqdm import tqdm

class CompatibilityTaskEngine:
    """
    Task: Outfit Compatibility Prediction
    Metric: AUC (Area Under Curve)
    Description: Binary classification task. 
                 Label 1 for compatible outfits (real), 
                 Label 0 for incompatible outfits (fake/shuffled).
    """
    
    @staticmethod
    def generate(outfit_df, vector_db, output_dir, split="test", pool_size=100):
        """
        Builds the Compatibility (AUC) test set.
        Args:
            pool_size: If -1, use random sampling. 
                       If > 0, use Hard Negative search via vector_db.
        """
        # 1. Filter by split
        target_df = outfit_df[outfit_df['split'] == split]
        if target_df.empty:
            print(f"⚠️ Warning: Split '{split}' is empty or not found.")
            return

        tasks = []
        for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc=f"Generating Compatibility Tasks [{split}]"):
            item_idxs = list(row['item_indices']) # Positive sample
            
            # --- A. Save Positive Sample ---
            tasks.append({
                "items": [int(idx) for idx in item_idxs],
                "label": 1
            })
            
            # --- B. Generate Negative Sample ---
            target_pos = random.choice(range(len(item_idxs)))
            gt_item_idx = item_idxs[target_pos]
            
            # Strategy selection
            if pool_size == -1:
                # Simple Random Negative: random choice from the whole library
                all_ids = vector_db.item_df['item_idx'].values
                neg_item_idx = random.choice(all_ids)
            else:
                # Hard Negative: find visually similar items that don't belong here
                gt_emb = vector_db.get_embeddings(gt_item_idx)
                # Search for more candidates to ensure we can filter out the GT
                search_results = vector_db.search(gt_emb, k=pool_size + 1)
                
                # Filter out the ground truth itself
                neg_pool = [res[0] for res in search_results if int(res[0]) != int(gt_item_idx)]
                
                if not neg_pool: # Fallback if search fails
                    neg_item_idx = random.choice(vector_db.item_df['item_idx'].values)
                else:
                    neg_item_idx = random.choice(neg_pool)
            
            # Create the 'fake' outfit
            neg_outfit = item_idxs.copy()
            neg_outfit[target_pos] = int(neg_item_idx)
            
            tasks.append({
                "items": [int(idx) for idx in neg_outfit],
                "label": 0
            })

        # --- C. Save to JSON ---
        os.makedirs(output_dir, exist_ok=True)
        # Use a descriptive name including pool_size to distinguish strategies
        output_path = os.path.join(output_dir, f"compatibility_{split}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2)
            
        print(f"✅ Compatibility Task saved to {output_path}")

    @staticmethod
    def load(file_path):
        """
        Loads the Compatibility task file.
        Returns a DataFrame for easier metric calculation.
        """
        pass