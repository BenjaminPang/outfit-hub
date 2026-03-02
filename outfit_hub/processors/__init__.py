from .base_processor import BaseProcessor
from .ifashion_processor import iFashionProcessor
from .polyvoreu_processor import PolvyoreUProcessor
from .fashion32_processor import Fashion32Processor
from .polyvore_outfits_processor import PolyvoreOutfitsProcessor


# 建立 名字 -> 类的映射表
PROCESSOR_MAP = {
    "ifashion": iFashionProcessor,
    "polyvoreu": PolvyoreUProcessor,
    "fashion32": Fashion32Processor,
    "polyvore_outfits": PolyvoreOutfitsProcessor,
}


def get_processor(dataset_name, manager, **kwargs):
    if dataset_name not in PROCESSOR_MAP:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Check PROCESSOR_MAP.")
    
    processor_class = PROCESSOR_MAP[dataset_name]
    return processor_class(dataset_name=dataset_name, manager=manager, **kwargs)