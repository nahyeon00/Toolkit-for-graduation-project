# Open Intent Classification (WIP)


## Key features
- [**Transformers**](https://https://huggingface.co/docs/transformers/index) 
- [**Lightning**](https://lightning.ai//) 


## 실행 방법1
config.yaml에 모델, 데이터, Trainer를 지정
```bash
python train.py --config <config.yaml>
```

실행 예제
```bash
python train.py --config samples/feature_extractor.yaml
```

## 실행 방법2
모델, 데이터, Trainer를 각각 따로 지정
``bash
python main.py fit --model <model-yaml> --trainer <trainer-yaml> --data <data-yaml>
```

실행 예제
```bash
python main.py fit --model samples/fe.yaml --trainer samples/trainer.yaml --data samples/stackvoerflow.yaml 
```

