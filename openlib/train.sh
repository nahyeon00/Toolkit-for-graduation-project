#!/bin/bash

export TOKENIZERS_PARALLELISM=false

# samples/config.yaml == pretraining

python train.py --config samples/KNN_bert-base-uncased_stackoverflow_0.25_1_output.yaml  --seed 0

# python train.py --config samples/K_1_way.yaml  --seed 0

