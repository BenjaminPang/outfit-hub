import os
import io
from abc import ABC, abstractmethod
import hashlib
import json
from typing import List, Dict, Tuple, Optional
import tarfile
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm
import pickle

from ..utils.image_utils import process_and_pad_image
from ..utils.clip_utils import ClipEmbedding


class BaseProcessor(ABC):
    item_dtype_dict = {
        'item_id': 'string',
        'item_idx': 'int',
        'category_id': 'string',
        'category_idx': 'int',
        'source': 'string',
    }
    outfit_dtype_dict = {
        'outfit_id': 'string',
        'outfit_idx': 'int',
        'item_ids': 'object',
        'item_indices': 'object',
        'length': 'int',
        'source': 'string',
        'split': 'string',
    }
    user_dtype_dict = {
        'user_id': 'string',
        'user_idx': 'int',
        'outfit_indices': 'object',
        'outfit_num': 'int',
        'source': 'string',
    }

    def __init__(self, dataset_name, dataset_config, img_size=224, chunk_size=50000):
        self.dataset_name = dataset_name
        self.datast_config = dataset_config
        self.img_size = img_size
        self.chunk_size = chunk_size
        self.category_len = 0
        self.supported_tasks = {}
        
        # Load paths from registry
        self.root_path = dataset_config['root_path']
        self.image_dir = dataset_config['image_dir']

        self.output_path = dataset_config.get('output_path', os.path.join("./data", self.dataset_name))
        os.makedirs(self.output_path, exist_ok=True)

        self.itemid2itemidx = {}

        self.item_parquet = []
        self.outfit_parquet = []
        self.user_parquet = []

    @abstractmethod
    def process_category(self):
        """
        Every dataset has its own category mapping strategy. In order to unified each category, we need to transform them into unique category idx and store the idx: string mapping in {output_path}/category.json
        """
        pass

    @abstractmethod
    def parse_raw_data(self):
        """
        This method is the main process. This process the relationship of the original outfits, items, and users.
        First, find out all meet-requirement outfit, give metadata to outfit.
        Second, find out all unique items used in these valid outfits
        Third, give idx to each item and save item_indices (List of item idx) into outfit metadata
        Fourth, extract item clip embedding
        Fifth, find out all user relationship to outfit (Optional)
        """
        pass

    def save_parquet(self):
        """
        The processed metadata list is converted into a DataFrame, and then the data is forcibly converted to the defined dtype and saved as Parquet.
        """
        print(f"--- Saving Parquet files to {self.output_path} ---")
        
        # 定义保存任务：(数据列表, 类型字典, 文件名)
        tasks = [
            (self.item_parquet, self.item_dtype_dict, 'items.parquet'),
            (self.outfit_parquet, self.outfit_dtype_dict, 'outfits.parquet'),
            (self.user_parquet, self.user_dtype_dict, 'users.parquet')
        ]

        for data_list, dtype_dict, file_name in tasks:
            if not data_list:
                print(f"⚠️ Warning: {file_name} data is empty, skipping.")
                continue

            df = pd.DataFrame(data_list)

            for col, dtype in dtype_dict.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except Exception as e:
                        print(f"❌ Error converting column {col} to {dtype}: {e}")

            if file_name == 'outfits.parquet' and 'split' in df.columns:
                print(f"\n📊 Dataset Split Summary ({self.dataset_name}):")
                counts = df['split'].value_counts()
                total = len(df)
                for split_name, count in counts.items():
                    percentage = (count / total) * 100
                    print(f"  - {split_name:7}: {count:6d} ({percentage:6.2f}%)")
                print("-" * 30)

            # 3. 保存为 Parquet
            save_file_path = os.path.join(self.output_path, file_name)
            df.to_parquet(save_file_path, index=False)
            print(f"✅ Saved {len(df)} rows to {save_file_path}")

    def save_tar(self):
        """
        Divide items into chunks and save them into multiple .tar files.
        This prevents creating excessively large single archives.
        """
        tar_idx = 0
        for i in range(0, len(self.item_parquet), self.chunk_size):
            items_for_current_tar = self.item_parquet[i : i + self.chunk_size]
            tar_path = os.path.join(self.output_path, f"{tar_idx:03d}.tar")
            
            print(f"创建新 Tar 包: {tar_path} (包含 {len(items_for_current_tar)} 张)")
            self._parallel_save_to_tar(tar_path, items_for_current_tar, max_workers=8, target_size=self.img_size)
            
            if len(items_for_current_tar) == self.chunk_size:
                tar_idx += 1

    def process_clip(self, batch_size: int = 4096):
        """
        Extracts only CLIP image features from preprocessed TAR files.
        Saves a dictionary: {item_idx: image_embedding_numpy} to a PKL file.
        """
        print(f"--- Extracting Image CLIP Features: {self.dataset_name} ---")
        
        clip_tool = ClipEmbedding()
        image_features_dict = {}

        # Iterate through each TAR chunk
        current_tar_idx = 0
        while True:
            tar_name = f"{current_tar_idx:03d}.tar"
            tar_path = os.path.join(self.output_path, tar_name)
            
            if not os.path.exists(tar_path):
                break # No more chunks to process

            print(f"📦 Processing: {tar_name}")
            
            with tarfile.open(tar_path, "r") as tar:
                # Filter only image files (ends with .jpg)
                members = [m for m in tar.getmembers() if m.name.endswith(".jpg")]
                
                # Process in batches
                for b_idx in tqdm(range(0, len(members), batch_size), desc="CLIP Vision Encoding", leave=False):
                    batch_members = members[b_idx : b_idx + batch_size]
                    
                    batch_bytes = []
                    batch_idxs = []
                    
                    for member in batch_members:
                        # Parse item_idx from filename "123.jpg"
                        idx = int(member.name.split('.')[0])
                        img_file = tar.extractfile(member)
                        
                        if img_file:
                            batch_bytes.append(img_file.read())
                            batch_idxs.append(idx)
                    
                    if not batch_bytes:
                        continue
                    
                    # Inference: Image only
                    img_embs = clip_tool.get_image_features(batch_bytes)
                    
                    # Update main dictionary
                    for j, idx in enumerate(batch_idxs):
                        image_features_dict[idx] = img_embs[j]
            
            current_tar_idx += 1

        # Save to PKL
        save_path = os.path.join(self.output_path, 'clip_vision_features.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(list(image_features_dict.values()), f)
        
        print(f"Success: Processed {len(image_features_dict)} images. Saved to: {save_path}")

    @abstractmethod
    def process_test(self):
        """
        Every children class should implement this method to generate corresponding test file from raw test data.
        """

    def save_metadata(self):
        """
        Summarizes dataset statistics and saves them to data/metadata.json.
        """
        print(f"--- Generating Metadata for {self.dataset_name} ---")
        
        # 转换成 DataFrame 方便统计
        item_df = pd.DataFrame(self.item_parquet)
        outfit_df = pd.DataFrame(self.outfit_parquet)
        user_df = pd.DataFrame(self.user_parquet)

        save_path = os.path.join(os.path.dirname(self.output_path), 'metadata.json')
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                stats = json.load(f)
        else:
            stats = dict()

        stats[self.dataset_name] = {
            "counts": {
                "items": len(item_df),
                "outfits": len(outfit_df),
                "users": len(user_df),
                "used_categories": item_df['category_idx'].nunique() if 'category_idx' in item_df.columns else 0,
                "raw_categories": self.category_len,
            },
            "splits": outfit_df['split'].value_counts().to_dict() if 'split' in outfit_df.columns else {},
            "avg_outfit_length": float(outfit_df['length'].mean()) if 'length' in outfit_df.columns else 0,
            "max_outfit_length": int(outfit_df['length'].max()) if 'length' in outfit_df.columns else 0,
            "image_size": self.img_size,
            "supported_tasks": self.supported_tasks,
            "chunk_size": self.chunk_size,
            "last_processed": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 路径处理：建议保存在每个数据集的根目录下
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Metadata saved to {save_path}")
        return stats

    def run(self, stage):
        """
        Main execution flow controlled by stages.
        Stage 1: Raw data ingestion, item indexing, image packaging, and CLIP extraction.
        Stage 2: Evaluation task generation and metadata summary.
        """
        print(f"--- Processing {self.dataset_name} | Stage: {stage} ---")

        if stage == 1:
            # 阶段 1：数据清洗与特征提取
            self.process_category()
            self.parse_raw_data()
            self.save_parquet()  # transform metadata and embedding into parquet and save to output_path
            self.save_tar()  # save preprocessed image file to output_path
            self.process_clip()  # Extract clip feature from image file
            
        elif stage == 2:
            # 阶段 2：评测任务生成
            if not self.item_parquet or not self.outfit_parquet:
                self.load_processed_data()
                
            self.process_test()  # output json to output_path
            self.save_metadata()  # save summarized dataset info into data/metadata.json
        else:
            raise ValueError(f"Invalid stage: {stage}. Use 1 (Ingestion) or 2 (Evaluation).")

    def load_processed_data(self):
        """
        Load the processed Parquet file from disk into memory.
        For use in the second phase (test generation) standalone runtime.
        """
        print(f"--- Loading processed data for {self.dataset_name} ---")
        with open(os.path.join(self.output_path, 'category.json'), 'r') as f:
            self.idx2category = json.load(f)
        self.category2idx = {v: k for k, v in self.idx2category.items()}
        self.category_len = len(self.idx2category)

        item_path = os.path.join(self.output_path, 'items.parquet')
        outfit_path = os.path.join(self.output_path, 'outfits.parquet')
        user_path = os.path.join(self.output_path, 'users.parquet')

        if os.path.exists(item_path):
            self.item_df = pd.read_parquet(item_path)
            self.item_parquet = self.item_df.to_dict('records')
            self.itemid2itemidx = self.item_df.set_index('item_id')['item_idx'].to_dict()
        if os.path.exists(outfit_path):
            self.outfit_df = pd.read_parquet(outfit_path)
            self.outfit_parquet = self.outfit_df.to_dict('records')
        if os.path.exists(user_path):
            self.user_df = pd.read_parquet(user_path)
            self.user_parquet = self.user_df.to_dict('records')
            self.userid2useridx = self.user_df.set_index('user_id')['user_idx'].to_dict()
        
        print(f"✅ Loaded: {len(self.item_parquet)} items, {len(self.outfit_parquet)} outfits.")

    def _parallel_save_to_tar(self, tar_path: str, items: List[Dict], max_workers: int = 8, target_size: int = 291, bg_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Processes images in parallel and packages them images into a .tar file.
        
        Args:
            tar_path (str): Path to the output tar file.
            max_workers (int): Number of parallel processes for image transformation.
            target_size (int): The dimension (width/height) for the square output image.
            bg_color (tuple): RGB tuple for padding background color.
        """
        # Always use 'w' (write) mode to create a new tar file. If the file already exists, it will be overwritten.
        mode = "w"
        
        # Use partial to 'freeze' static arguments (target_size, bg_color) 
        # allowing the map function to pass only the unique 'item' dictionary.
        worker_func = partial(self._worker_process_image, target_size=target_size, bg_color=bg_color)
        
        with tarfile.open(tar_path, mode) as tar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map the worker function across the items list. 
                # chunksize=100 reduces inter-process communication overhead for small tasks.
                results = executor.map(worker_func, items, chunksize=100)
                
                # Iterate through results as they are completed
                pbar = tqdm(results, total=len(items), desc=f"Packaging to {os.path.basename(tar_path)}", leave=False)
                for result in pbar:
                    if result:
                        filename, img_bytes = result
                        
                        # Create TarInfo object for the file entry
                        info = tarfile.TarInfo(name=filename)
                        info.size = len(img_bytes)
                        
                        # Write the byte stream into the tar archive
                        tar.addfile(tarinfo=info, fileobj=io.BytesIO(img_bytes))

    @staticmethod
    def _worker_process_image(item: Dict[str, str], target_size: int, bg_color: Tuple[int, int, int]) -> Optional[Tuple[str, bytes]]:
        """
        Isolated worker logic to process a single image. 
        Designed as a staticmethod to prevent pickling 'self' during multiprocessing.
        
        Returns:
            A tuple of (filename, image_bytes) if successful, else None.
        """
        # Execute the core image transformation logic (resizing, padding, etc.)
        img_data = process_and_pad_image(
            item.get('ori_path', ''), 
            target_size=target_size, 
            bg_color=bg_color
        )
        
        if img_data:
            # Construct a consistent filename using the item index
            return f"{item['item_idx']}.jpg", img_data
            
        return None


    @staticmethod
    def generate_outfit_id(item_ids: list[str]) -> tuple[List, str]:
        """生成outfit的唯一ID
        Args:
            items: item id的列表
        Returns:
            outfit的唯一ID
        """
        # 对items进行排序
        sorted_items = sorted(item_ids)
        # 用逗号分隔符将items连接起来
        text = ','.join(sorted_items)
        outfit_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        return sorted_items, outfit_id
