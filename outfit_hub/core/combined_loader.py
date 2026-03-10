from torch.utils.data import ConcatDataset, DataLoader

from .loader import OutfitTrainDataset


def get_combined_loader(dataset_configs, split='train', batch_size=32, **kwargs):
    """
    dataset_configs: List[dict], such as:
    [
        {"name": "polyvoreu519", "load_img": True},
        {"name": "ifashion", "load_img": True}
    ]
    """
    datasets = []
    
    for config in dataset_configs:
        ds = OutfitTrainDataset(
            root_dir="data", 
            dataset_name=config['name'], 
            split=split,
            load_img=config.get('load_img', False),
            **kwargs
        )
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)
    
    return DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'),
        collate_fn=OutfitTrainDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )