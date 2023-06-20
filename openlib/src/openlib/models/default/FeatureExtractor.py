from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics

from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor


class FeatureExtractor(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.3,
        num_classes: int = None,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
        freeze=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerFeatureExtractor(model_name_or_path, dropout_prob=dropout_prob)

        # if freeze:
        #     self.model = self.freeze_bert_parameters(self.model)
        self.classifier = nn.Linear(self.model.dense.out_features, num_classes)
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, batch):
        outputs = self.model(batch)
        logits = self.classifier(outputs)
        return logits

    def training_step(self, batch, batch_idx):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        logits = self.forward(model_input)
        labels = batch["labels"]

        loss = F.cross_entropy(logits, labels.long().squeeze(-1))
        self.log_dict({"loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_input = {k: v for k, v in batch.items() if k != "labels"}
        logits = self.forward(model_input)

        labels = batch["labels"]
        preds = logits.argmax(dim=-1)

        val_acc = self.metric(preds, labels)
        self.log("val_acc", val_acc)
        return val_acc

    def on_validation_epoch_end(self):
        self.log("val_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()

    def test_step(self, batch, batch_nb):
        labels = batch["labels"]

        model_input = {k: v for k, v in batch.items() if k != "labels"}
        logits = self.forward(model_input)
        preds = logits.argmax(dim=-1)

        test_acc = self.metric(preds, labels)  # check once again
        self.log("test_acc", test_acc)
        return {"test_acc": test_acc}


    def on_test_epoch_end(self):
        self.log("test_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.hparams.lr)

        warm_up_steps = int(self.trainer.estimated_stepping_batches * 0.1)
        scheduler = get_scheduler(
            self.hparams.scheduler_type,
            optimizer,
            # num_warmup_steps=self.hparams.warmup_steps,
            num_warmup_steps=warm_up_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def freeze_bert_parameters(self, model):
        print("freeze_bert_para")
        for name, param in model.model.named_parameters():
            param.requires_grad = False
            # print('name', name)
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
        return model
