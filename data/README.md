# 👕 Outfit Hub Datasets

This directory contains standardized fashion outfit datasets processed by **Outfit Hub Repository**: [BenjaminPang/outfit-hub](https://github.com/BenjaminPang/outfit-hub.git). For more information, please refer to this repository.

**Hugging Face Repository**: [pangkaicheng/outfit-hub-datasets](https://huggingface.co/datasets/pangkaicheng/outfit-hub-datasets)

---

## 📂 Supported Datasets

The following datasets have been standardized and uploaded:

| Dataset Name | Description | Key Features |
| :--- | :--- | :--- |
| `polyvore_outfits_disjoint` | Polyvore Dataset | Items in test set do not overlap with training set. |
| `polyvore_outfits_nondisjoint` | Polyvore Dataset | Standard split with potential item overlap. |
| `polyvoreu519` | Polyvore-U (User) | 519 unique users with personalized outfit sequences. |
| `polyvoreu630` | Polyvore-U (User) | 630 unique users with personalized outfit sequences. |
| `ifashion` | Large-scale E-commerce | Massive real-world outfit data from Alibaba/iFashion. |
| `fashion32` | Fine-grained Outfits | Highly categorized outfits for specific style matching. |

---

## 🛠 Unified Directory Structure

To ensure cross-dataset compatibility, every dataset folder follows this strict schema:

```text
dataset_name/
├── items.parquet             # Item metadata (idx, id, category.)
├── outfits.parquet           # Outfit definitions (item_idx lists, splits)
├── category.json             # Global category mapping (cate_idx -> Cate_str)
├── clip_vision_features.pkl  # Pre-computed CLIP-ViT-B/32 image features
├── [000-N].tar               # Sharded image archives (Tar format for high IO)
└── eval/                     # Standardized Evaluation Tasks
    ├── compatibility_*.json  # Compatibility Prediction (Binary/Score)
    └── fitb_*.json           # Fill-In-The-Blank (Multiple Choice)
```
---

## 📊 Data Schema Reference

### 1. items.parquet
* item_idx (int): Unified auto-increment index used by DataLoaders.
* item_id (string): Original ID from the raw dataset.
* category_idx (int): Standardized category ID.
* category_id (int): Original category ID.
* category (string): Original category in String.
* ori_path (string): Relative path inside the Tar shards or local disk.
* source (string): Dataset name

### 2. outfits.parquet
* outfit_idx (int): Unique identifier for the outfit.
* outfit_id (string): MD5 Hash generated from sorted item_ids, providing a globally unique identifier.
* item_ids (list[string]): List of original item IDs in the outfit.
* item_indices (list[int]): List of item_idx mapping to items.parquet.
* length (int): Number of items in the outfit.
* source (string): Origin dataset name (e.g., ifashion).
* split (string): Dataset split (train, valid, or test).

### 3. users.parquet (optional)
* user_id (string): Original user id from raw dataset.
* user_idx (int): Unified auto-increment index for users.
* outfit_indices (list[int]): List of outfit_idx mapping to outfits.parquet.
* outfit_num (int): Number of outfits created by this user.
* source (string): Dataset name.