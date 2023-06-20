import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


from src.openlib.models.knncl.base_knncl import Base_KNNCL
from src.openlib.models.knncl.knncl_util import generate_positive_sample, _prepare_inputs

import os
import pickle

class KNNCL(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        lof_path = None,
        tsne_path = None,
        label=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.unseen_label_id = num_classes  # should be fix

        self.model = Base_KNNCL(
            model_name_or_path=model_name_or_path, num_classes=num_classes
        )
        self.metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes + 1  # +1: unknown
        )
        self.train_total_features = torch.empty((0, self.model.encoder_q.config.hidden_size))
        self.test_total_features = torch.empty((0, self.model.encoder_q.config.hidden_size))

        self.lof = LocalOutlierFactor(n_neighbors=20, contamination = 0.05, novelty=True, n_jobs=-1)

        self.total_labels = torch.empty(0, dtype=torch.long)
        self.pooled_output = torch.empty((0, self.model.encoder_q.config.hidden_size))
        self.total_y_pred = torch.empty(0,dtype=torch.long)

        self.label_to_id = label
        ##persona
        if lof_path != None:
            self.lof_name = lof_path
        
        if tsne_path != None:
            self.tsne_path = tsne_path
    
    def forward(self, batch, positive_sample=None, negative_sample=None):
        if positive_sample != None:
            output = self.model(batch, positive_sample, negative_sample)
            return output
        else:
            probs, seq_embed = self.model(batch)
            return probs, seq_embed
    def on_fit_start(self) -> None:
        """Initialize the centroid."""
        self.create_negative_dataset(
            data_loader=self.trainer.datamodule.train_dataloader()
        )  # type: ignore

    def training_step(self, batch, batch_idx):
        device = next(self.model.parameters()).device
        labels = batch["labels"]
        positive_sample = None
        positive_sample = generate_positive_sample(self.negative_dataset, labels)
        positive_sample = _prepare_inputs(device, positive_sample)
        # print("forward")

        loss = self.forward(batch, positive_sample=positive_sample)
        self.log("loss", loss[0], prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # print("\nvalidation\n")
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]

        probs, seq_embed = self.forward(model_input)
        preds = torch.argmax(probs, dim=1)
        
        val_acc = self.metric(preds, labels)
        self.log("val_acc", val_acc)
        return val_acc

    def on_validation_epoch_end(self):
        self.log("val_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()

    
    def on_train_end(self):
        # print("\none_test_start") 
        device = next(self.model.parameters()).device
        self.model.eval()
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination = 0.05, novelty=True, n_jobs=-1)
        for batch in tqdm(self.trainer.datamodule.train_dataloader()):
            labels = batch['labels']
            model_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.set_grad_enabled(False):
                output = self.model(model_input)
                self.train_total_features = torch.cat((self.train_total_features.to(device), output[1]))
        self.train_total_features = self.train_total_features.detach().cpu().numpy()
        self.lof.fit(self.train_total_features)
        with open(self.lof_name, 'wb') as file:
            pickle.dump(self.lof, file)

    def on_test_start(self) -> None:
        with open(self.lof_name, 'rb') as file:
            self.lof = pickle.load(file)
        return super().on_test_start()

    def test_step(self, batch, batch_nb):
        # print("test_step")
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        outputs = self.forward(model_input)

        total_prob, y_pred = outputs[0].max(dim=1)
        y_pred = y_pred.cpu().numpy()
        
        feats = outputs[1].cpu().numpy()

        # lof 로 예측하기
        y_pred_lof = pd.Series(self.lof.predict(feats))  # -1로 예측한 이상치 데이터의 인덱스 추출

        y_pred[y_pred_lof[y_pred_lof == -1].index] = self.unseen_label_id  # 특정 레이블로 할당함
        y_pred = torch.tensor(y_pred)
        test_acc = self.metric(y_pred.detach().cpu(), labels.detach().cpu())

        self.pooled_output = torch.cat((self.pooled_output.to(self.device), outputs[1]))
        self.total_labels = torch.cat((self.total_labels.to(self.device), labels))
    
        self.total_y_pred = torch.cat((self.total_y_pred.to(self.device), y_pred.to(self.device)))

        self.log("test_acc", test_acc, prog_bar=True)

    def on_test_epoch_end(self):
        pooled_output = self.pooled_output.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()
        y_pred = self.total_y_pred.cpu().numpy()
        self.draw_label(pooled_output, y_true)

        # confusion matrix
        label_mapping = {value: key for key, value in self.label_to_id.items()}
        labels = list(label_mapping.values())
        labels.append('open')
        # breakpoint()
        self.plot_confusion_matrix(y_true,y_pred,labels)

        scalar_label = labels[int(self.total_y_pred[-1].item())]
        print("prediction", scalar_label)

        with open('output.txt', 'w') as file:
            file.write(str(scalar_label))

        self.log("test_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def create_negative_dataset(self, data_loader: DataLoader) -> None:
        """create negative dataset."""
        self.negative_dataset = {}
        device = next(self.model.parameters()).device
        for batch in tqdm(data_loader):
            labels = batch['labels']
            neg_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            for i, label in enumerate(labels):
                label = int(label)
                if label not in self.negative_dataset.keys():
                    self.negative_dataset[label] = [{key: value[i] for key, value in neg_inputs.items()}]
                else:
                    self.negative_dataset[label].append({key: value[i] for key, value in neg_inputs.items()})

    def draw_label(self, weights, labels):
        print("TSNE: fitting start...")
        tsne = TSNE(n_components=2, metric='cosine', random_state=0, n_jobs=4)
        # n_components: Dimension of the embedded space    # n_job: the number of CPU core to run parallel
        embedding = tsne.fit_transform(weights)

        df = pd.DataFrame(embedding, columns=['x', 'y'])  # x, y 값을 DataFrame으로 변환
        df['label'] = labels  # 라벨을 DataFrame에 추가
        df.to_csv(self.tsne_path + '.csv', index=False)
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, normalize=False, title=None):
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            xlabel='Predicted label',
            ylabel='True label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        if title:
            ax.set_title(title)

        fig.tight_layout()
        plt.savefig(self.tsne_path + '.pdf')
        plt.close(fig)
 