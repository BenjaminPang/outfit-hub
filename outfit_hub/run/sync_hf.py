# scripts/sync_hf.py
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from outfit_hub.utils.sync_hf_utils import download_repo, upload_repo


def get_hf_token():
    """统一从 .env 加载 Token"""
    env_path = Path.cwd() / '.env'
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"💡 已从 {env_path.absolute()} 加载环境配置")
    else:
        print(f"⚠️ 未找到 .env 文件，尝试读取系统环境变量...")

    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    if not token:
        raise ValueError("❌ 错误: 未在 .env 或环境变量中找到 HUGGINGFACE_TOKEN")
    
    return token


def main():
    parser = argparse.ArgumentParser(description="Hugging Face 双向同步工具 (Model & Dataset)")
    # 将 type 提取到全局，或者在子命令中分别添加
    subparsers = parser.add_subparsers(dest="action", help="选择操作: upload 或 download")

    # --- 共同参数模版 ---
    def add_common_args(sub_parser):
        sub_parser.add_argument("--repo", type=str, required=True, help="仓库 ID (如 user/repo)")
        sub_parser.add_argument("--path", type=str, required=True, help="本地路径")
        sub_parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="HF Token")
        sub_parser.add_argument("--type", type=str, choices=["model", "dataset"], default="model", help="仓库类型: model 或 dataset")

    # --- Upload 子命令 ---
    up_parser = subparsers.add_parser("upload", help="上传本地文件夹到 HF")
    up_parser.add_argument("--files", nargs="+", help="指定上传的文件列表(支持多个)")
    add_common_args(up_parser)

    # --- Download 子命令 ---
    dl_parser = subparsers.add_parser("download", help="从 HF 下载到本地")
    add_common_args(dl_parser)
    dl_parser.add_argument("--no-symlinks", action="store_true", help="强制下载真实文件而非软链接")

    args = parser.parse_args()

    if not args.token:
        try:
            token = get_hf_token()
        except Exception as e:
            print(f"❌ 错误: 未提供 Token。请通过 --token 传入或设置 HF_TOKEN 环境变量。 {e}")
            sys.exit(1)
    else:
        token = args.token

    if args.action == "upload":
        upload_repo(args.repo, args.type, args.path, token, files=args.files)
    elif args.action == "download":
        download_repo(args.repo, args.type, args.path, token, no_symlinks=args.no_symlinks)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()