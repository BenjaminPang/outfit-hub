# utils/vector_db_utils.py
from typing import List, Dict
from tqdm import tqdm

import chromadb
import numpy as np


class VectorDB:
    def __init__(self, items_df, collection_name, root='.', persistent=True):
        # 内存中维护一套 ID 和 Embedding 的映射
        self.items_df = items_df
        self.collection_name = collection_name

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

    def search(self, query_emb, k):
        """利用 ChromaDB 进行向量搜索"""
        query_list = query_emb.reshape(1, -1).tolist()
        res = self.collection.query(query_embeddings=query_list, n_results=k)
        
        # 将结果转回 (item_id, similarity)
        results = []
        for iid, dist, meta in zip(res['ids'][0], res['distances'][0], res['metadatas'][0]):
            results.append((int(iid), 1.0 - float(dist), meta))  # range [-1, 1], 越大越相似
        return results

    def update_features(self, item_idxs: list[int], embeddings: List[np.ndarray], metadatas: Dict):
        """
        批量更新或插入特征到 ChromaDB
        item_idxs: 原始整数索引列表
        embeddings: 特征矩阵 List[Dim]
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
        
    def get_embedding_by_idx(self, item_idx: int) -> np.ndarray:
        """根据原始整数索引获取对应的向量"""
        # 1. 将整数索引转为字符串 ID
        id_str = str(item_idx)
        
        # 2. 调用 get 方法，并明确指定需要获取 embeddings
        res = self.collection.get(
            ids=[id_str], 
            include=["embeddings"]
        )
        
        # 3. 检查结果并返回
        if res['embeddings'] is not None:
            # 返回的是一个 list，我们取第一个元素并转回 numpy 数组
            return res['embeddings'][0]
        else:
            return None
    
    def __len__(self):
        return self.collection.count()

    def clear_collection(self):
        all_data = self.collection.get()
        if all_data['ids']:
            self.collection.delete(ids=all_data['ids'])
            print(f"{len(all_data)} item information in {self.collection.name} deleted")
