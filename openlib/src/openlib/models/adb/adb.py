from collections import defaultdict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor

from .boundaryloss import BoundaryLoss
from transformers import AutoTokenizer



def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


class ADBModel(nn.Module):
    """Implementation of ADB model"""

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
    ):
        super().__init__()

        self.feature_extractor = TransformerFeatureExtractor(model_name_or_path)

        self.num_classes = num_classes
        self.unseen_label_id = num_classes  # should be fix

        feature_output_dim = self.feature_extractor.dense.out_features
        self.register_buffer("centroids", torch.zeros(self.num_classes, feature_output_dim))
        self.centroids: Tensor
        #         self.delta = nn.Parameter(torch.randn(num_classes))  # learnable delta
        self.delta = nn.Parameter(
            torch.abs(torch.randn(num_classes))
        )  # 강제로 delta 양수로 initialization
        # nn.init.normal_(self.delta)

    def initialize_centroid(self, data_loader: DataLoader) -> None:
        """Initialize the centroid."""

        device = next(self.feature_extractor.parameters()).device
        self.feature_extractor.eval()
        with torch.no_grad():
            outputs = []
            for batch in tqdm(data_loader):
                model_input = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                pooled_output = self.feature_extractor(model_input)
                labels = batch["labels"]

                for l, output in zip(labels, pooled_output):
                    self.centroids[l] += output
                outputs.append({"label": labels})

            labels = torch.cat([x["label"] for x in outputs]).detach().to(device)
            self.centroids /= torch.bincount(labels).float().unsqueeze(1)
        self.feature_extractor.train()

    def forward(self, batch):
        if self.centroids is None:
            raise ValueError(
                "Centroids are not initialized. Run `initialize_centroid` method first."
            )

        with torch.no_grad():
            pooled_output = self.feature_extractor(batch)

        return pooled_output

    def open_classify(self, features):
        logits = euclidean_metric(features, self.centroids)
        _, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        boundary = F.softplus(self.delta)[preds]
        preds[euc_dis >= boundary] = self.unseen_label_id
        return preds


class ADB(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        freeze=True,
        tsne_path=None,
        label=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ADBModel(model_name_or_path=model_name_or_path, num_classes=num_classes)
        self.metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes + 1  # +1: unknown
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.criterion = BoundaryLoss()
        self.total_labels = torch.empty(0, dtype=torch.long)
        self.pooled_output = torch.empty((0, self.model.feature_extractor.model.config.hidden_size))
        self.total_y_pred = torch.empty(0,dtype=torch.long)

        self.label_to_id = label
        if tsne_path != None:
            self.tsne_path = tsne_path

    def forward(self, batch):
        pooled_output = self.model(batch)  # call forward in feature_extractor
        return pooled_output

    def on_fit_start(self) -> None:
        """Initialize the centroid."""
        self.model.initialize_centroid(
            data_loader=self.trainer.datamodule.train_dataloader()
        )  # type: ignore

    def training_step(self, batch, batch_idx):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        pooled_output = self.forward(model_input)
        labels = batch["labels"]

        loss = self.criterion(pooled_output, self.model.centroids, self.model.delta, labels)
        self.log_dict({"loss": loss}, prog_bar=True)
        return loss

    # def on_validation_start(self):
    #     print(self.model.centroids)
    #     print(self.model.delta)

    def validation_step(self, batch, batch_idx):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        pooled_output = self.forward(model_input)
        preds = self.model.open_classify(pooled_output)
        labels = batch["labels"]
        loss = self.criterion(pooled_output, self.model.centroids, self.model.delta, labels)
        val_acc = self.metric(preds, labels)
        self.log_dict({"val_acc": val_acc, "val_loss": loss})

    def on_validation_epoch_end(self):
        self.log("val_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()

    def test_step(self, batch, batch_nb):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        pooled_output = self.forward(model_input)
        labels = batch["labels"]

        preds = self.model.open_classify(pooled_output)
        test_acc = self.metric(preds, labels)

        self.pooled_output = torch.cat((self.pooled_output.to(self.device), pooled_output))
        self.total_labels = torch.cat((self.total_labels.to(self.device), labels))

        self.total_y_pred = torch.cat((self.total_y_pred.to(self.device), preds))

        self.log("test_acc", test_acc)

    def on_test_epoch_end(self):
        pooled_output = self.pooled_output.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()
        y_pred = self.total_y_pred.cpu().numpy()
        self.draw_label(pooled_output, y_true)
        # confusion matrix
        # label_to_id = self.trainer.datamodule.label_to_id
        label_mapping = {value: key for key, value in self.label_to_id.items()}
        labels = list(label_mapping.values())
        labels.append('open')
        self.plot_confusion_matrix(y_true,y_pred,labels)

        scalar_label = labels[int(self.total_y_pred[-1].item())]
        print("prediction", scalar_label)

        with open('output.txt', 'w') as file:
            file.write(str(scalar_label))

        self.log("test_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()

    def predict_step(self, batch, batch_nb):
        self.eval()
        model_input = {k: v.squeeze(0) for k, v in batch.items()}
        with torch.no_grad():
            pooled_output = self.forward(model_input)
            preds = self.model.open_classify(pooled_output)
        return preds

    ####### delta만 optimize & warmup proportion
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW([self.model.delta], lr=self.hparams.lr)
        # self.model.delta.parameter()
        optimizer = torch.optim.Adam([self.model.delta], lr=self.hparams.lr)
        # print("op", optimizer.param_groups[0]['params'])
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def draw_label(self, weights, labels):
        print("TSNE: fitting start...")
        tsne = TSNE(n_components=2, metric='cosine', random_state=0, n_jobs=4)
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
