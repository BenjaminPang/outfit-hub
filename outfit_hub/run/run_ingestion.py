# outfit_hub/run/run_ingestion.py
import yaml

from outfit_hub.processors import get_processor


def main():
    with open("outfit_hub/registry.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Supported dataset name: [ifashion, polyvoreu519, polyvoreu630, fashion32, polyvore_outfits_disjoint, polyvore_outfits_nondisjoint]
    dataset_name = "polyvore_outfits_nondisjoint"
    proc = get_processor(
        dataset_name=dataset_name, 
        dataset_config=config[dataset_name],
        img_size=291
    )
    
    # proc.run(stage=1)
    proc.run(stage=2)


if __name__ == "__main__":
    main()