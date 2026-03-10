import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi


def upload_data_to_hf(repo_id, local_data_path):
    # 1. load token from .env
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("Error: HUGGINGFACE_TOKEN not found in .env file")

    api = HfApi()

    # 2. 确保仓库存在 (如果不存在则创建)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            exist_ok=True
        )
        print(f"✅ 目标仓库已就绪: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️ 创建仓库时出错 (可能已存在): {e}")

    # 3. 上传整个 data 文件夹
    print(f"🚀 开始上传 {local_data_path} 目录下的所有数据...")
    
    try:
        api.upload_folder(
            folder_path=local_data_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            # 建议在 HF 上直接放在根目录，或者指定 path_in_repo="data"
            path_in_repo=".", 
            # 过滤掉不需要的文件（可选）
            ignore_patterns=["*.pyc", "__pycache__", ".DS_Store", '.jpg', '.png'],
            commit_message=f"Update README.md"
        )
        print("🎉 上传成功！")
    except Exception as e:
        print(f"❌ 上传失败: {e}")


if __name__ == "__main__":
    # 配置区
    # 请将 '你的用户名/数据集名称' 修改为你实际的 HF 路径
    USER_REPO_ID = "pangkaicheng/outfit-hub-datasets"
    
    # 自动定位根目录下的 data 文件夹
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_PATH = BASE_DIR / "data"

    if DATA_PATH.exists():
        upload_data_to_hf(USER_REPO_ID, str(DATA_PATH))
    else:
        print(f"❌ 错误: 找不到数据目录 {DATA_PATH}")