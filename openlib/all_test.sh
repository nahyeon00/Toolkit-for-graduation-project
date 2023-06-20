#!/bin/bash

export TOKENIZERS_PARALLELISM=false

model_name_or_path=$1
export model_name_or_path="bert-base-uncased"  # 변수 받는거($1) make_config.py에 변수 받을 거 적어둠


# 전체 학습된 모델로 test만 진행 -> 얘도 따로 yaml 파일 만들어야함 일단 냅두기
#기존 모델 바로 가지고올 수 있는 방법 고려
python /workspace/openlib/model_test.py --config /workspace/openlib/samples/adb.yaml
