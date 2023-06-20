#!/bin/bash

export TOKENIZERS_PARALLELISM=false


export method=$1
export model_name_or_path=$2
export dataset=$3
export known_cls_ratio=$4
export ip=$5
export max_epoch=$6

python k_1_personalize.py --config samples/$1_$2_$3_$4_$6_output.yaml  --seed 0

