#!/bin/bash

export TOKENIZERS_PARALLELISM=false

# 변수 받기 
#model_name_or_path=$1
#dataset=$2
#known_cls_ratio=$3

# 파이썬 파일로 변수 넘겨주기
export model_name_or_path=$1  # 변수 받는거($1) make_config.py에 변수 받을 거 적어둠
export dataset=$2
export known_cls_ratio=$3


# output.yaml 파일 만들기
python make_config.py

# 만들어진 output.yaml 파일로 feature 학습
python openlib/fe_train.py --config openlib/samples/config.yaml

# python make_config2.py  ##

# 전체 학습 -> 얘도 따로 yaml 파일 만들어야함 일단 냅두기
python openlib/train.py --config openlib/samples/adb.yaml
