from .core.train_dataset import OutfitTrainDataset, get_combined_loader
from .core.eval_dataset import FITBEvalDataset, CompEvalDataset

# 导出版本信息（可选）
__version__ = "0.0.1"
__all__ = [
    "get_combined_loader",
    "OutfitTrainDataset",
    "FITBEvalDataset",
    "CompEvalDataset"
]