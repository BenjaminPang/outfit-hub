# 👕 Outfit Hub Datasets

This directory contains standardized fashion outfit datasets processed by **Outfit Hub Repository**: [BenjaminPang/outfit-hub](https://github.com/BenjaminPang/outfit-hub.git). For more information, please refer to this repository.

**Hugging Face Repository**: [pangkaicheng/outfit-hub-datasets](https://huggingface.co/datasets/pangkaicheng/outfit-hub-datasets)

## 🚀 Data Preparation

### 1. Setup
First, clone the repository and install it as a package.
```bash
git clone https://github.com/BenjaminPang/outfit-hub.git
cd outfit_hub
pip install -e .
```

### 2. Download Pre-processed Datasets
Currently, all datasets listed in the Supported Datasets section have been standardized and are hosted on Hugging Face. You can download them directly using the built-in script:

Modify the download section in ./outfit_hub/run/sync_hf.sh to ensure the path points to your local data directory, then run:

```Bash
python3 outfit_hub/run/sync_hf.py download \
    --repo pangkaicheng/outfit-hub-datasets \
    --path ./data \
    --type dataset \
    --no-symlinks
```

This command will automatically download all .parquet metadata, .npy vision features, and sharded image .tar archives from Hugging Face.


### 3. Processing from Raw Data (Ingestion)
If you possess the raw datasets and wish to run the standardization pipeline yourself, use `run_ingestion.py` to control the processing stages:

```Python
# Set the dataset_name in outfit_hub/run/run_ingestion.py
dataset_name = "DATASET_NAME"

# Run specific stages
proc.run(stage=1)  # Stage 1: Data cleaning, indexing, image packaging, and CLIP extraction
proc.run(stage=2)  # Stage 2: Evaluation task generation (FITB/Compatibility) and metadata summary
```

Stage 1: The core flow includes process_category (category unification), parse_raw_data (index building), save_parquet (metadata storage), save_tar (image packaging), and process_clip (vision feature extraction).

Stage 2: Focuses on evaluation logic, including process_test (generating standardized test sets) and save_metadata (updating global metadata.json).

### 4. Extension: Implementing New Datasets
To support a new dataset, you need to inherit from the BaseProcessor class and override the following three key abstract methods:

* `process_category(self)`: Handle raw category strings, establish `idx: string` mappings, and save them to `category.json`.

* `parse_raw_data(self)`: Establish relationships between Users, Outfits and Items, generating metadata lists that conform to the schema.

* `process_test(self)`: Convert raw test data into the unified `eval/*.json` format (e.g., Compatibility Prediction and FITB tasks).

### 5. Image Extraction and Access
For high-efficiency IO with large-scale data, images are stored in sharded .tar archives by default. If you need direct access to .jpg files for training process, it is recommended to extract all images for the current dataset:

```Bash
chmod +x ./outfit_hub/run/extract_tar.sh
./outfit_hub/run/extract_tar.sh
```

Extract a specific Tar archive to a target directory:

```Bash
tar -xvf data/dataset_name/000.tar -C data/dataset_name/
```


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