from torch.utils.data import ConcatDataset, DataLoader


def get_combined_loader(dataset_names, dataset_class, root_dir="./data", split='train', batch_size=32, num_workers=4, max_seq_length=9, pin_memory=True, transform=None, load_clip=True, load_img=False):
    """
    dataset_configs: List[str], such as:
    ["polyvoreu519", "ifashion"]
    """
    datasets = []
    for ds_idx, name in enumerate(dataset_names):
        ds = dataset_class(
            root_dir=root_dir, 
            dataset_name=name, 
            dataset_idx=ds_idx,
            split=split,
            max_seq_length=max_seq_length,
            transform=transform,
            load_clip=load_clip,
            load_img=load_img,
        )
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets)
    
    return DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=(split == "train"),
        collate_fn=dataset_class.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )