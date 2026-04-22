# utils/vector_db_utils.py
import os
from typing import Union
from tqdm import tqdm

import pandas as pd
from PIL import Image
import chromadb
import numpy as np


class VectorDB:
    def __init__(self, collection, feature_path):
        """
        Unified Vector Storage and Indexing Handle.
        
        This class manages two synchronized storage layers:
        1. ChromaDB (HNSW Index): For efficient Top-K similarity search.
        2. Numpy Memmap (Flat Storage): For zero-copy, multi-process feature retrieval.
        """
        self.collection = collection
        self.feature_path = feature_path
        self._embedding_memmap = None

    @classmethod
    def setup_and_sync(cls, cfg, dataset_name, collection_name, encode_fn):
        """
        Args:
            cfg (dict): config dict.
            dataset_name (str): Name of desired dataset.
            collection_name (str): Unique identifier for the specific encoder/version. {dataset_name}__{encode_name}__{model_name}__{version}
            encode_fn (callable, optional): Model inference function. Only required in this function.
        """
        root = cfg.train.get("data_root_dir", './data')
        db_path = os.path.join(root, "vector_db")
        client = chromadb.PersistentClient(path=db_path)
        dist_mode = "cosine" if "clip" in collection_name.lower() else "l2"
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": dist_mode}
        )

        feature_dir = os.path.join(db_path, "feature")
        os.makedirs(feature_dir, exist_ok=True)
        feature_path = os.path.join(feature_dir, f"feat_{collection_name}.npy")

        items_df = pd.read_parquet(os.path.join(root, dataset_name, "items.parquet"))
        if not os.path.exists(feature_path):
            batch_size = 5000
            all_idxs = items_df.index.tolist()
            all_embs_list = []
            
            for i in tqdm(range(0, len(all_idxs), batch_size), desc="Encoding"):
                batch_idxs = all_idxs[i : i + batch_size]
                imgs = cls.get_image_by_idx(os.path.join(root, dataset_name), batch_idxs)
                txts = items_df.loc[batch_idxs, 'description'].fillna('').tolist()
                
                embs = encode_fn(imgs, txts)
                metadatas = items_df.loc[batch_idxs].to_dict(orient='records')
                
                # 1. 更新 ChromaDB 索引
                collection.upsert(
                    ids=[str(i) for i in batch_idxs],
                    embeddings=embs,
                    metadatas=metadatas
                )
                all_embs_list.append(embs)

            # 2. 汇总并持久化到 .npy 文件 (核心点)
            full_matrix = np.vstack(all_embs_list).astype(np.float32)
            np.save(feature_path, full_matrix)
            print(f"Memmap file saved at {feature_path}")
        
        full_matrix = np.load(feature_path, mmap_mode='r') # 使用 mmap 节省内存
        expected_count = len(items_df)
        if collection.count() < expected_count:
            print(f"⚠️ ChromaDB sync issue: Found {collection.count()}/{expected_count} items. Re-syncing from .npy...")
            
            batch_size = 5000
            for i in tqdm(range(0, expected_count, batch_size), desc="Restoring ChromaDB"):
                end_idx = min(i + batch_size, expected_count)
                batch_idxs = list(range(i, end_idx))
                
                # 从 mmap 中切片拿到 embeddings
                embs = full_matrix[i:end_idx].tolist()
                
                # 从 items_df 拿到对应的元数据
                # 注意：这里假设 items_df 的索引与 .npy 的行一一对应
                metadatas = items_df.iloc[batch_idxs].to_dict(orient='records')
                
                collection.upsert(
                    ids=[str(idx) for idx in batch_idxs],
                    embeddings=embs,
                    metadatas=metadatas
                )
            print(f"✅ Re-sync complete. ChromaDB now has {collection.count()} items.")
        else:
            print(f"✨ ChromaDB and .npy are in sync ({collection.count()} items).")


        return cls(collection, feature_path)

    @classmethod
    def create_lazy_reader(cls, cfg, collection_name):
        root = cfg.train.data_root_dir
        db_path = os.path.join(root, "vector_db")
        client = chromadb.PersistentClient(path=db_path)
        dist_mode = "cosine" if "clip" in collection_name.lower() else "l2"
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": dist_mode}
        )

        feature_path = os.path.join(db_path, "feature", f"feat_{collection_name}.npy")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"{feature_path} is not found")

        return cls(collection, feature_path)
    
    @property
    def embedding_cache(self):
        """核心：通过 memmap 实现多进程零拷贝共享"""
        if self._embedding_memmap is None:
            if not os.path.exists(self.feature_path):
                raise FileNotFoundError(f"Feature file {self.feature_path} missing. Run sync first.")
            # 'r' 模式确保多进程只读共享且物理内存唯一
            self._embedding_memmap = np.load(self.feature_path, mmap_mode='r')
        return self._embedding_memmap
    
    def search(self, query_emb: np.ndarray, k: int):
        """利用 ChromaDB 进行向量搜索"""
        query_list = query_emb.reshape(1, -1)  # use (n_queries, dim) for robotness
        res = self.collection.query(query_embeddings=query_list, n_results=k)
        
        # 将结果转回 (item_id, similarity)
        results = []
        for iid, dist, meta in zip(res['ids'][0], res['distances'][0], res['metadatas'][0]):
            if self.collection.metadata['hnsw:space'] == 'cosine':
                sim = 1.0 - float(dist) / 2  # range (0, 1), 1 means most closed
            elif self.collection.metadata['hnsw:space'] == 'l2':
                sim = 1.0 - float(dist) / 4.0  # range (0, 1), 1 means most closed
            else:
                sim = 1.0 - float(dist)
            results.append((int(iid), sim, meta))  # range [0, 1], 越大越相似
        return results
    
    def get_nearest_neighbors_ids(self, item_idx: int, k: int) -> list[int]:
        """
        给定一个物品索引，从 DB 中找到最相似的 k 个物品 ID
        """
        # 1. 先拿到该单品的 Embedding
        target_emb = self.get_embedding_by_idx(item_idx)
        if target_emb is None:
            return []
            
        # 2. 搜索 top-k (通常搜 k+1，因为最相似的往往是它自己)
        search_res = self.search(target_emb, k + 1)
        
        # 3. 提取 ID，并排除掉自己（如果存在的话）
        neighbor_ids = []
        for iid, sim, meta in search_res:
            if iid != item_idx:
                neighbor_ids.append(iid)
        
        # 返回前 k 个（确保长度）
        return neighbor_ids[:k]
        
    def get_embedding_by_idx(self, item_idx: Union[int, list[int]]) -> np.ndarray:
        """
        Get embeddings using numpy indexing.
        - int: returns (dim,)
        - list/array: returns (n, dim)
        """
        return self.embedding_cache[item_idx]
    
    @staticmethod
    def get_image_by_idx(dataset_dir, batch_idxs: list[int]) -> list[Image.Image]:
        images = []
        for item_idx in batch_idxs:
            img_path = os.path.join(dataset_dir, 'images', f"{item_idx}.jpg")
            
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to load {img_path}, error: {e}")
                    images.append(None)
            else:
                images.append(None)
                
        return images
    
    def __len__(self):
        return self.collection.count()

    def clear_collection(self):
        all_data = self.collection.get()
        if all_data['ids']:
            self.collection.delete(ids=all_data['ids'])
            print(f"{len(all_data)} item information in {self.collection.name} deleted")
        if os.path.exists(self.feature_path):
            os.remove(self.feature_path)
