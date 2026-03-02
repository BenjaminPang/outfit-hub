import json
import os
import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor


class Fashion32Processor(BaseProcessor):
    def process_category(self):
        category_list = ['all']
        self.category2idx = {category_str: i for i, category_str in enumerate(category_list)}
        self.idx2category = {i: category_str for i, category_str in enumerate(category_list)}
        self.category_len = len(category_list)

        save_path = os.path.join(self.output_path, 'category.json')
        with open(save_path, 'w') as f:
            json.dump(self.idx2category, f, indent=2)

    def process_outfits(self):
        # Implementation for mapping outfit item_ids to global_ids
        pass