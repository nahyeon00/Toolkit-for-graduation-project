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

# echo "done"
start_time=$(date +%s.%N)

# output.yaml 파일 만들기
python /workspace/openlib/make_config.py &&
# echo "make_config.py"

# 만들어진 output.yaml 파일로 feature 학습
python /workspace/openlib/fe_train.py --config /workspace/openlib/samples/$1_$2_$3_$4_$5_output.yaml --seed 0 &&
# echo "fe_train.py"

python /workspace/openlib/make_model.py &&

# 전체 학습 -> 얘도 따로 yaml 파일 만들어야함 일단 냅두기
python /workspace/openlib/train.py --config /workspace/openlib/samples/$1_$2_$3_$4_$5_output1.yaml  --seed 0;
# echo "train.py"

# 종료 시간 측정
end_time=$(date +%s.%N)

# 소요 시간 계산
elapsed_time=$(echo "$end_time - $start_time" | bc)

echo "ADB : $elapsed_time 초"
