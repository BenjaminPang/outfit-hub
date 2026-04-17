# utils/vector_db_utils.py
import os
from typing import Union
from tqdm import tqdm

from PIL import Image
import chromadb
import numpy as np


class VectorDB:
    def __init__(self, items_df, collection_name, encode_fn, root='.', persistent=True, force_recompute=False):
        # 内存中维护一套 ID 和 Embedding 的映射
        self.items_df = items_df
        self.collection_name = collection_name
        self.encode_fn = encode_fn
        self.root = root

        # 初始化 ChromaDB (仅用于 top-k 相似度检索)
        if persistent:
            db_path = f"{root}/vector_db"
            print(f"📦 Using Persistent Storage at: {db_path}")
            self.client = chromadb.PersistentClient(path=db_path)
        else:
            print("🚀 Using Ephemeral Storage (In-Memory)")
            self.client = chromadb.EphemeralClient()

        dist_mode = "cosine" if "clip" in collection_name else "l2"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": dist_mode}
        )
        self.sync_embeddings(force_recompute)
        self._embedding_cache = self._load_vector_db_to_numpy()

    def sync_embeddings(self, force_recompute=False):
        """
        Core Management Method:
            - If `force_recompute=True`, clear the current Collection and recompute.
            - Otherwise, only fill in the missing features.
        """
        batch_size=1000
        if force_recompute or self.collection.count() < len(self.items_df):
            if self.encode_fn is None:
                raise ValueError("Need encode_fn to sync embeddings.")

            print(f"⚠️ Warning: Recomputing ALL embeddings for {self.collection_name}")
            self.clear_collection() 

            all_idxs = self.items_df.index.tolist()
            if all_idxs:
                for i in tqdm(range(0, len(all_idxs), batch_size), desc="Syncing DB"):
                    batch_idxs = all_idxs[i : i + batch_size]

                    imgs = self.get_image_by_idx(batch_idxs)
                    txts = self.items_df.loc[batch_idxs, 'description'].fillna('').tolist()
                    
                    embs = self.encode_fn(imgs, txts)
                    metadatas = self.items_df.loc[batch_idxs].to_dict(orient='records')
                    self.update_features(batch_idxs, embs, metadatas)

    def _load_vector_db_to_numpy(self):
        res = self.collection.get(include=['embeddings'])
        
        dim = len(res['embeddings'][0])
        all_embs = np.zeros((len(self.items_df), dim), dtype=np.float32)
        
        ids = np.array(res['ids'], dtype=int)
        embs = np.array(res['embeddings'], dtype=np.float32)
        
        all_embs[ids] = embs 
        return all_embs

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

    def update_features(self, item_idxs: list[int], embeddings: list[np.ndarray], metadatas: dict):
        """
        批量更新或插入特征到 ChromaDB
        item_idxs: 原始整数索引列表
        embeddings: 特征矩阵 list[Dim]
        """
        if len(item_idxs) == 0:
            return

        ids_str = [str(i) for i in item_idxs]
        try:
            self.collection.upsert(
                ids=ids_str,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"❌ Error during VectorDB upsert: {e}")
            raise e
        
    def get_embedding_by_idx(self, item_idx: Union[int, list[int]]) -> np.ndarray:
        return self._embedding_cache[item_idx]
        
    def get_image_by_idx(self, batch_idxs: list[int]) -> list[Image.Image]:
        items_entries = self.items_df.iloc[batch_idxs]
        images = []
        for entry in items_entries.itertuples():
            # 获取数据（根据你的列名访问，例如 entry.source）
            dataset_name = entry.source
            item_idx = entry.Index  # 或者 entry.id，取决于你的列名
            img_path = os.path.join(self.root, dataset_name, 'images', f"{item_idx}.jpg")
            
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


class StyleFeatureVectorDB(VectorDB):
    def __init__(self, parent_vector_db, collection_name, encode_fn, persistent=True, force_recompute=False):
        self.parent_vector_db = parent_vector_db
        super().__init__(parent_vector_db.items_df, collection_name, encode_fn, root=parent_vector_db.root, persistent=persistent, force_recompute=force_recompute)

    def sync_embeddings(self, force_recompute=False):
        """
        Core Management Method:
            - If `force_recompute=True`, clear the current Collection and recompute.
            - Otherwise, only fill in the missing features.
        """
        batch_size=1000
        if force_recompute or self.collection.count() < len(self.items_df):
            if self.encode_fn is None:
                raise ValueError("Need encode_fn to sync embeddings.")

            print(f"⚠️ Warning: Recomputing ALL embeddings for {self.collection_name}")
            self.clear_collection() 

            all_idxs = self.items_df.index.tolist()
            if all_idxs:
                for i in tqdm(range(0, len(all_idxs), batch_size), desc="Syncing DB"):
                    batch_idxs = all_idxs[i : i + batch_size]
                    features = self.parent_vector_db.get_embedding_by_idx(batch_idxs)
                    embs = self.encode_fn(features)
                    metadatas = self.items_df.loc[batch_idxs].to_dict(orient='records')
                    self.update_features(batch_idxs, embs, metadatas)
