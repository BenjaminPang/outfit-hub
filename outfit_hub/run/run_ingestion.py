# outfit_hub/run/run_ingestion.py
from outfit_hub.core.manager import DatasetManager
from outfit_hub.processors import get_processor


def main():
    # 1. Init ID Manager
    manager = DatasetManager(config_path="outfit_hub/registry.yaml")
    
    # 2. Run iFashion Processor
    # Supported dataset name: [ifashion, polyvoreu, fashion32, polyvore_outfits]
    proc = get_processor(
        dataset_name="polyvore_outfits", 
        manager=manager,
        img_size=(224, 224)
    )
    
    proc.run()


if __name__ == "__main__":
    main()