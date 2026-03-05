# outfit_hub/run/run_ingestion.py
from outfit_hub.core.manager import DatasetManager
from outfit_hub.processors import get_processor


def main():
    # 1. Init ID Manager
    manager = DatasetManager(config_path="outfit_hub/registry.yaml")
    
    # 2. Run iFashion Processor
    # Supported dataset name: [ifashion, polyvoreu, fashion32, polyvore_outfits_disjoint, polyvore_outfits_nondisjoint]
    dataset_name = "ifashion"
    proc = get_processor(
        dataset_name=dataset_name, 
        dataset_config=manager.config[dataset_name],
        img_size=291
    )
    
    proc.run(stage=1)


if __name__ == "__main__":
    main()