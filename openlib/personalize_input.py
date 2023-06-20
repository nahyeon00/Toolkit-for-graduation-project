from argparse import ArgumentParser, Namespace

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from src.openlib.config import get_configurable_parameters
from src.openlib.data import get_datamodule
from src.openlib.models import get_model

from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from src.openlib.models.adb.adb import ADB

import os
import yaml
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

method = os.environ.get('method')
model_name_or_path = os.environ.get('model_name_or_path')  # bert-base-uncased
dataset1 = os.environ.get('dataset')  #stackoverflow
known_cls_ratio = os.environ.get('known_cls_ratio')
ip = os.environ.get('ip')
max_epoch = os.environ.get('max_epoch')

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to a overall config file")
    parser.add_argument("--model", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--data", type=str, required=False, help="Path to a data config file")
    parser.add_argument("--trainer", type=str, required=False, help="Path to a trainer config file")
    parser.add_argument("--seed", type=str, required=False, help="Path to a trainer config file")

    args = parser.parse_args()

    return args


def get_callbacks(callback_list):
    callbacks = []
    for callback in callback_list:
        callback_args = {
            callback["class_path"].split(".")[-1]: callback.get("init_args", None)
        }

        # TODO: too naive :(
        if "EarlyStopping" in callback_args:
            early_stopping = EarlyStopping(**callback_args["EarlyStopping"])
            callbacks.append(early_stopping)

        if "ModelCheckpoint" in callback_args:
            model_checkpoint = ModelCheckpoint(**callback_args["ModelCheckpoint"])
            callbacks.append(model_checkpoint)

        if "RichProgressBar" in callback_args:
            progress_bar = RichProgressBar()
            callbacks.append(progress_bar)

    return callbacks


def train():
    args = get_args()
    config = get_configurable_parameters(args=args)

    if args.seed is not None:
        seed_everything(args.seed)
    else:
        seed_everything(1235, workers=True)

    sentence = str(ip)

    #test data에 input값 추가
    data_path = os.path.join(config.data.data_path, config.data.dataset)
    test_data_path = f"{data_path}/test.tsv"
    test_df = pd.read_csv(test_data_path,  delimiter='\t')
    new_data = {
            'text': [sentence],
            'label': ['None']
    }
    new_df = pd.DataFrame(new_data, index=[0])
    test_df = pd.concat([test_df, new_df], ignore_index=True)
    test_df.to_csv(test_data_path, sep='\t', index=False, mode='w')

    datamodule = get_datamodule(config)
    datamodule.prepare_data()
    label_to_id = datamodule.label_to_id
    datamodule.setup(stage='test')

    config.model.init_args.num_classes = datamodule.num_classes

    path = 'outputs/ADB_'+str(model_name_or_path)+'_'+str(dataset1)+'_'+str(known_cls_ratio)+'_'+max_epoch+'_1.ckpt'  # ckpt
    # path = '/workspace/openlib/outputs/adb.ckpt'
    model = ADB.load_from_checkpoint(path, num_classes=datamodule.num_classes, label=label_to_id)  # 마지막에 저장된 ckpt
    callbacks = get_callbacks(config.trainer.pop("callbacks", []))

    trainer = Trainer(**config.trainer, callbacks=callbacks)

    # predictions = trainer.predict(model, dataloaders=[dataloader])
    test_dataloader = datamodule.test_dataloader()
    # pred = trainer.test(model, dataloaders=test_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    print("tttttt test end")

    tsne_df = pd.read_csv(config.model.init_args.tsne_path+'.csv')
    label_mapping = {v: k for k, v in label_to_id.items()}
    tsne_df['label'] = tsne_df['label'].map(label_mapping).fillna('오픈')
    ori_df = pd.read_csv(datamodule.test_data_path, sep="\t")
    tsne_df['sentence'] = ori_df['text']

    # test data 속 input 삭제
    test_df = test_df.drop(test_df.index[-1])  # test data 속 input 삭제
    test_df.to_csv(test_data_path, sep='\t', index=False, mode='w')

    tsne_df.to_csv(config.model.init_args.tsne_path+'.csv', index=False)
    print("end")


if __name__ == "__main__":
    train()



