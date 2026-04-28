from torch.utils.data import ConcatDataset, DataLoader

from ..utils.vector_db_utils import VectorDB


def get_combined_loader(dataset_names, dataset_class, vector_db_dict, root_dir="./data", split='train', batch_size=32, num_workers=4, pin_memory=True, transform=None):
    """
    dataset_configs: List[str], such as:
    ["polyvoreu519", "ifashion"]
    """
    datasets = []
    for ds_idx, name in enumerate(dataset_names):
        ds = dataset_class(
            root_dir=root_dir,
            feature_path=vector_db_dict[name].feature_path,
            dataset_name=name, 
            dataset_idx=ds_idx,
            split=split,
            transform=transform,
        )
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)
    
    return DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=(split == "train"),
        collate_fn=dataset_class.collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=pin_memory
    )