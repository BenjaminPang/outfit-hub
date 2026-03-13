# outfit_hub/utils/sync_hf_utils.py
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
import sys


def download_repo(repo, type, path, token, no_symlinks=False):
    """
    高性能下载逻辑，支持 Model 和 Dataset
    """
    print(f"🔍 准备下载 [{type}]: {repo}")
    print(f"📂 目标本地路径: {path}")

    try:
        snapshot_download(
            repo_id=repo,
            repo_type=type,  # 动态设置: model 或 dataset
            local_dir=path,
            token=token,
            local_dir_use_symlinks=not no_symlinks, 
            ignore_patterns=[".*", "__pycache__", ".DS_Store", '.jpg', '.png', "temp_images"],
        )
        print(f"✅ 下载成功！已存至: {path}")
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        sys.exit(1)


def upload_repo(repo, type, path, token, files=None):
    api = HfApi(token=token)
    
    # 情况 A: 指定了特定文件列表
    if files:
        print(f"🎯 选定文件上传模式: {len(files)} 个文件")
        for file_pattern in files:
            # 在本地寻找匹配的文件
            # 注意：这里假设你在 checkpoints 根目录运行，或者传入相对路径
            local_file = os.path.join(path, file_pattern)
            if not os.path.exists(local_file):
                print(f"⚠️ 找不到文件: {local_file}, 跳过...")
                continue
            
            print(f"⬆️ 正在上传: {local_file} -> {repo}")
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=file_pattern, # 保持在仓库中的相对路径一致
                repo_id=repo,
                repo_type=type,
                token=token
            )
        print(f"✅ 选定文件全部上传完成！")
    
    # 情况 B: 整文件夹上传
    else:
        print(f"🚀 正在全量上传文件夹: {path}")
        api.upload_folder(
            folder_path=path,
            repo_id=repo,
            repo_type=type,
            ignore_patterns=[".*", "__pycache__", ".DS_Store", '*.jpg', '*.png', "temp_images/**", "*.bak", "vector_db/**"],
        )