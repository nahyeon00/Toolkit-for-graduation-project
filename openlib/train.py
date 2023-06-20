from argparse import ArgumentParser, Namespace

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from src.openlib.config import get_configurable_parameters
from src.openlib.data import get_datamodule
from src.openlib.models import get_model

import pandas as pd


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
        callback_args = {callback["class_path"].split(".")[-1]: callback.get("init_args", None)}

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

    datamodule = get_datamodule(config)
    datamodule.prepare_data()
    label_to_id = datamodule.label_to_id
    print("label_to_id",label_to_id)

    config.model.init_args.num_classes = datamodule.num_classes
    config.model.init_args.label = label_to_id
    model = get_model(config)
    callbacks = get_callbacks(config.trainer.pop("callbacks", []))

    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)

    # label 만 저장
    path = config.project.path
    df = pd.DataFrame(label_to_id, index = [0])
    df.to_csv(path)

    # tsne 위한 저장
    trainer.test(model=model, datamodule=datamodule)
    tsne_df = pd.read_csv(config.model.init_args.tsne_path+'.csv')
    label_mapping = {v: k for k, v in label_to_id.items()}
    tsne_df['label'] = tsne_df['label'].map(label_mapping).fillna('오픈')
    ori_df = pd.read_csv(datamodule.test_data_path, sep="\t")
    tsne_df['sentence'] = ori_df['text']

    tsne_df.to_csv(config.model.init_args.tsne_path+'.csv', index=False)


    


if __name__ == "__main__":
    train()
