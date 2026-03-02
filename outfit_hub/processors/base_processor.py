import os
from abc import ABC, abstractmethod

import pandas as pd
from PIL import Image
from tqdm import tqdm


class BaseProcessor(ABC):
    def __init__(self, dataset_name, manager, img_size=(224, 224)):
        self.dataset_name = dataset_name
        self.manager = manager
        self.img_size = img_size
        
        # Load paths from registry
        self.raw_path = manager.config[dataset_name]['raw_data_path']
        self.image_dir = manager.config[dataset_name]['image_dir']

        self.output_path = manager.config[dataset_name]['output_path']
        os.makedirs(self.output_path, exist_ok=True)

    @abstractmethod
    def process_category(self):
        """
        Every dataset has its own category mapping strategy. In order to unified each category, we need to transform them into unique category idx and store the idx: string mapping in {output_path}/category.json
        """
        pass


    def run(self):
        """Main execution flow."""
        print(f"--- Processing Dataset: {self.dataset_name} ---")
        self.process_category()
        # self.process_outfit()
        # self.process_item()
        # self.process_user()
        # self.process_parquet()  # transform metadata and embedding into parquet and save to output_path
        # self.process_tar()  # save preprocessed image file to output_path
        # self.process_test()  # output json to output_path
        # self.save_metadata()  # save summarized dataset info into data/metadata.json
