# outfit_hub/run/run_ingestion.py
import yaml
import argparse

from outfit_hub.processors import get_processor


def main():
    parser = argparse.ArgumentParser(description="Ingesting Dataset Framework")
    parser.add_argument("dataset_name", type=str, choices=["ifashion", "polyvoreu519", "polyvoreu630", "fashion32", "polyvore_outfits_disjoint", "polyvore_outfits_nondisjoint"])
    parser.add_argument("--img_size", type=int, default=291)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()

    with open("outfit_hub/registry.yaml", 'r') as f:
        config = yaml.safe_load(f)

    proc = get_processor(
        dataset_name=args.dataset_name, 
        dataset_config=config[args.dataset_name],
        img_size=args.img_size
    )
    
    if args.stage == 3:
        print(f"🚀 Running FULL pipeline for {args.dataset_name}")
        proc.run(stage=1)
        proc.run(stage=2)
    else:
        proc.run(stage=args.stage)


if __name__ == "__main__":
    main()
