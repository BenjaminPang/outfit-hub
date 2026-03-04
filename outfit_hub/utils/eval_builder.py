import random
import os
import json

from tqdm import tqdm


def build_fitb_test(vector_db, outfit_df, output_dir, split="train", pool_size=100):
    test_outfit_df = outfit_df[outfit_df['split'] == split]
    fitb_test = []
    # 3. 循环生成任务
    for _, row in tqdm(test_outfit_df.iterrows(), total=len(test_outfit_df), desc="Generating Hard-Negative FITB"):
        # outfit_list 格式: [user_id, item_idx_1, item_idx_2, item_idx_3, item_idx_4]
        outfit_list = row['item_indices'] 
        
        # 随机选一个位置作为“填空”考点
        target_pos = random.choice(range(len(outfit_list)))
        gt_item_idx = outfit_list[target_pos]
        
        # --- 寻找 Hard Negatives (语义相似但不同的单品) ---
        # 1. 获取正样本的 Embedding (内存直接读取)
        gt_emb = vector_db.get_embeddings(gt_item_idx)
        
        # 2. 在向量库中搜索最相似的 10 个
        # search 返回结果格式: [(item_idx, similarity, metadata), ...]
        search_results = vector_db.search(gt_emb, k=pool_size + 1)
        
        # 3. 过滤掉正样本本身，提取其 item_idx
        # 这里通过 int(gt_item_idx) 确保类型匹配
        neg_pool = [res[0] for res in search_results if int(res[0]) != int(gt_item_idx)]
        
        # 4. 从最相似的前 10 个候选中随机选 3 个作为干扰项
        if len(neg_pool) < 3:
            # 如果库中相似样本不足（理论上 16w 数据不会发生），随机抽样补齐
            neg_pool = random.sample(range(len(vector_db.item_df)), 3)
        
        selected_negs = random.sample(neg_pool[:pool_size], 3)
        
        # --- 步骤 C: 构造 4 个候选套装 ---
        candidates = []
        
        # 1. 第一个是 Ground Truth (强制要求)
        # 将 numpy 数组转回 list，并确保元素是 Python 原生 int 类型（方便 JSON 序列化）
        candidates.append([int(i) for i in outfit_list])
        
        # 2. 后三个是干扰项
        for neg_idx in selected_negs:
            temp_outfit = list(outfit_list)
            temp_outfit[target_pos] = int(neg_idx)
            candidates.append([int(i) for i in temp_outfit])
        
        # --- 步骤 D: 封装任务 ---
        fitb_test.append({
            "candidates": candidates,
            "gt_label": 0  # 因为我们手动把正确的放到了第一个
        })

    # 4. 保存 JSON
    output_path = os.path.join(output_dir, f"fitb_{split}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fitb_test, f, indent=4)

    print(f"\n✅ 成功生成 {len(fitb_test)} 条 Hard-Negative FITB 任务。")
    print(f"📂 文件路径: {output_path}")


def build_auc_testset(test_df, vector_db, pool_size=100):
    auc_tasks = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Building AUC Testset"):
        outfit = row['item_indices'] # 正样本
        
        # 1. 保存正样本
        auc_tasks.append({
            "items": [int(idx) for idx in outfit],
            "label": 1
        })
        
        # 2. 生成负样本 (替换其中一个位置)
        target_pos = random.choice(range(len(outfit)))
        
        # 策略：从全库随机抽一个作为干扰 (也可以用你的 search 找 Hard Negative)
        if pool_size == -1:
            neg_item_idx = random.choice(range(len(vector_db.item_df)))
        else:
            gt_item_idx = outfit[target_pos]
            gt_emb = vector_db.get_embeddings(gt_item_idx)
        
            # 2. 在向量库中搜索最相似的 10 个
            # search 返回结果格式: [(item_idx, similarity, metadata), ...]
            search_results = vector_db.search(gt_emb, k=pool_size + 1)
            
            # 3. 过滤掉正样本本身，提取其 item_idx
            # 这里通过 int(gt_item_idx) 确保类型匹配
            neg_pool = [res[0] for res in search_results if int(res[0]) != int(gt_item_idx)]
            neg_item_idx = random.choice(neg_pool)
        
        neg_outfit = outfit.copy()
        neg_outfit[target_pos] = int(neg_item_idx)
        
        auc_tasks.append({
            "items": [int(idx) for idx in neg_outfit],
            "label": 0
        })

    output_path = f"data/2_processed/auc_test_{pool_size}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(auc_tasks, f, indent=4)
