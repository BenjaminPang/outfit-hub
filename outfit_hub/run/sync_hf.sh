#!/bin/bash

python outfit_hub/run/sync_hf.py upload \
    --repo pangkaicheng/outfit-hub-datasets \
    --path ./data \
    --type dataset 

# python3 outfit_hub/run/sync_hf.py download \
#     --repo pangkaicheng/outfit-hub-datasets \
#     --path ./data \
#     --type dataset \
#     --no-symlinks