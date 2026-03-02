import json
import os
import csv
import pandas as pd
from tqdm import tqdm

from .base_processor import BaseProcessor


class PolyvoreOutfitsProcessor(BaseProcessor):
    def process_category(self):
        category_file_path = os.path.join(self.raw_path, "categories.csv")
        category_set = set()
        with open(category_file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # row 是一个列表，例如 ['11', 'sweater', 'tops']
                _, category1, category2 = row
                category_set.add(category1)
                category_set.add(category2)

        category_list = list(category_set)
        self.category2idx = {category_str: i for i, category_str in enumerate(category_list)}
        self.idx2category = {i: category_str for i, category_str in enumerate(category_list)}
        self.category_len = len(category_list)

        save_path = os.path.join(self.output_path, 'category.json')
        with open(save_path, 'w') as f:
            json.dump(self.idx2category, f, indent=2)

    def process_outfits(self):
        # Implementation for mapping outfit item_ids to global_ids
        pass