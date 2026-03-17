# utils/vector_db_utils.py
from typing import List
from tqdm import tqdm

import chromadb
import pandas as pd
import numpy as np


class VectorDB:
    def __init__(self, item_df, embeddings_raw, dataset_name, root='.'):
        # 内存中维护一套 ID 和 Embedding 的映射
        self.item_df = item_df
        self.embeddings_raw = embeddings_raw

        # 初始化 ChromaDB (仅用于 top-k 相似度检索)
        self.client = chromadb.PersistentClient(path=f"{root}/vector_db/{dataset_name}")
        self.collection = self.client.get_or_create_collection(
            name="items_catalog",
            metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0:
            print("💾 Initializing ChromaDB collection (first time)...")
            self._fill_chroma()

    def _fill_chroma(self):
        # Chroma 要求 ID 是字符串
        idxs = self.item_df['item_idx'].astype(str).tolist()
        metadatas = self.item_df.to_dict('records')
        
        batch_size = 5000
        for i in tqdm(range(0, len(idxs), batch_size)):
            self.collection.add(
                ids=idxs[i:i+batch_size],
                embeddings=self.embeddings_raw[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
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

    def get_embeddings(self, item_idxs: List[int]):
        """核心加速：直接从内存中的 NumPy 数组切片"""
        return self.embeddings_raw[item_idxs]
    
    def __len__(self):
        return self.collection.count()
