#!/bin/bash

export TOKENIZERS_PARALLELISM=false

# 변수 받기
# model_name_or_path=$1
# dataset=$2
# known_cls_ratio=$3

# 파이썬 파일로 변수 넘겨주기
export method=$1
export model_name_or_path=$2  # 변수 받는거($1) make_config.py에 변수 받을 거 적어둠
export dataset=$3
export known_cls_ratio=$4
export max_epoch=$5
#export method=$4

# echo "done"


# output.yaml 파일 만들기
python /workspace/openlib/K_make_config.py &&
# echo "make_config.py"

# 만들어진 output.yaml 파일로 feature 학습
#python /workspace/openlib/fe_train.py --config /workspace/openlib/samples/$4_$1_$2_$3_output.yaml &&
# echo "fe_train.py"

#python /workspace/openlib/make_model.py &&

# 전체 학습 -> 얘도 따로 yaml 파일 만들어야함 일단 냅두기
python /workspace/openlib/train.py --config /workspace/openlib/samples/$1_$2_$3_$4_$5_output.yaml  --seed 0;

