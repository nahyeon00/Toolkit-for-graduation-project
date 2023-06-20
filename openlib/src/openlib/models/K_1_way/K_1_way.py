import torch
from torch import Tensor
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from transformers import get_scheduler
import torchmetrics
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
# from src.openlib.models.K_1_way.base_k_1_way import Base_K_1

import os


class ConvexSampler(nn.Module):
    def __init__(self, feat_dim, unseen_label_id):
        super(ConvexSampler, self).__init__()
        self.multiple_convex = 1 #args.multiple_convex
        self.multiple_convex_eval = 1 # args.multiple_convex_eval
        self.unseen_label_id = unseen_label_id
        self.batch_size = 64  # config로 뱌꾸기
        self.oos_num = 64  # config로 바꾸기
        self.feat_dim = feat_dim


    def forward(self, z, label_ids, mode=None, device=None):
        num_convex = self.batch_size * self.multiple_convex
        num_convex_eval = self.batch_size * self.multiple_convex_eval
        convex_list = []
        if mode =='train':
            if label_ids.size(0)>2:
                # assert 1==0
                while len(convex_list) < num_convex:
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)   
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])
                convex_samples = torch.cat(convex_list, dim=0).view(num_convex, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat(
                    (label_ids.cpu(), torch.tensor([self.unseen_label_id] * num_convex).cpu()),
                    dim=0
                )
        elif mode == 'eval':
            if label_ids.size(0) > 2:
                val_num = num_convex_eval
                while len(convex_list) < val_num:
                    cdt = np.random.choice(label_ids.size(0), 2, replace=False)
                    if label_ids[cdt[0]] != label_ids[cdt[1]]:
                        s = np.random.uniform(0, 1, 1)
                        convex_list.append(s[0] * z[cdt[0]] + (1 - s[0]) * z[cdt[1]])
                convex_samples = torch.cat(convex_list, dim=0).view(val_num, -1)
                z = torch.cat((z, convex_samples), dim=0)
                label_ids = torch.cat(
                    (label_ids.cpu(), torch.tensor([self.unseen_label_id] * val_num).cpu()),
                    dim=0
                )
        return z, label_ids



class K_1_way(pl.LightningModule):
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
        tsne_path=None,
        label=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.unseen_label_id = num_classes  # should be fix

        self.model = self.initialize_feature_extractor(model_name_or_path)
        self.dense = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.activation = nn.ReLU()  #  nn.LeakyReLU(),
        self.dropout = nn.Dropout(dropout_prob)
        self.sampler = ConvexSampler(self.model.config.hidden_size, num_classes)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes + 1)
        self.total_labels = torch.empty(0, dtype=torch.long)
        self.pooled_output = torch.empty((0, self.model.config.hidden_size))
        self.total_y_pred = torch.empty(0,dtype=torch.long)

        self.metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes + 1  # +1: unknown
        )
        self.loss_fct = nn.CrossEntropyLoss()
        self.label_to_id = label
        self.t = 0.1 #hyperpararmeter args.temp
        if tsne_path != None:
            self.tsne_path = tsne_path

    def initialize_feature_extractor(self, model_name_or_path: str):
        if model_name_or_path.endswith("ckpt"):
            pretrained = torch.load(model_name_or_path)
            model_name_or_path = pretrained["hyper_parameters"]["model_name_or_path"]
            state_dict = pretrained["state_dict"]
            for key in list(state_dict.keys()):
                state_dict[key.replace("model.model.", "")] = state_dict.pop(key)

            model = AutoModel.from_pretrained(model_name_or_path, state_dict=state_dict)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path)
            model = AutoModel.from_pretrained(model_name_or_path, config=config)

        return model
    
    def forward(self, batch, mode=None, sampler=None, device=None, pred=None):
        if pred == None:
            model_input = {k: v for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]

            outputs = self.model(**model_input)
            # mean_pooling = outputs.last_hidden_state
            mean_pooling = outputs.last_hidden_state.mean(dim=1)   # bert[1] [bst, max_seq, 768]
            pooled_output = self.dense(mean_pooling)
            
            if sampler != None:  # tain, valid
                pooled_output, labels = self.sampler(pooled_output, labels, mode=mode, device=device)  # [128,768], [128]
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            return pooled_output, logits, labels
        else:  # predict step
            outputs = self.model(**batch)
            # mean_pooling = outputs.last_hidden_state
            mean_pooling = outputs.last_hidden_state.mean(dim=1)   # bert[1] [bst, max_seq, 768]
            pooled_output = self.dense(mean_pooling)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            return pooled_output, logits

    def training_step(self, batch, batch_idx):
        device = next(self.model.parameters()).device

        pos_batch = batch['pos']
        neg_batch = batch['neg']
        
        pos_pooled_output, pos_logits, pos_labels = self.forward(pos_batch, mode='train', sampler=True)  # sampler 진행
        neg_pooled_output, neg_logits, neg_labels = self.forward(neg_batch)  # sampler 진행 x

        pos_loss = self.loss_fct(torch.div(pos_logits, self.t).cpu(),pos_labels)
        neg_loss = self.loss_fct(torch.div(neg_logits, self.t).cpu(),neg_labels.cpu())

        loss = pos_loss + neg_loss
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        device = next(self.model.parameters()).device
        pooled_output, logits, labels = self.forward(batch, mode='eval', sampler=True)
        probs = F.softmax(logits, dim=1)
        _, total_preds = probs.max(dim = 1)
        val_acc = self.metric(total_preds, labels.to(device)).to(device)
        self.log("val_acc", val_acc)
        return val_acc

    def on_validation_epoch_end(self):
        self.log("val_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()

    def test_step(self, batch, batch_nb):
        device = next(self.model.parameters()).device
        pooled_output, logits, labels = self.forward(batch)
        probs = F.softmax(logits, dim=1)
        _, preds = probs.max(dim = 1)

        test_acc = self.metric(preds, labels.to(device)).to(device)

        self.pooled_output = torch.cat((self.pooled_output.to(self.device), pooled_output))
        self.total_labels = torch.cat((self.total_labels.to(self.device), labels))

        self.total_y_pred = torch.cat((self.total_y_pred.to(self.device), preds))

        self.log("test_acc", test_acc, prog_bar=True)
        return {"test_acc": test_acc}

    def on_test_epoch_end(self):
        pooled_output = self.pooled_output.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()
        y_pred = self.total_y_pred.cpu().numpy()
        self.draw_label(pooled_output, y_true)
        # confusion matrix
        label_mapping = {value: key for key, value in self.label_to_id.items()}
        labels = list(label_mapping.values())
        labels.append('open')
        self.plot_confusion_matrix(y_true,y_pred,labels)

        scalar_label = labels[int(self.total_y_pred[-1].item())]

        with open('output.txt', 'w') as file:
            file.write(str(scalar_label))

        self.log("test_acc", self.metric.compute(), prog_bar=True)
        self.metric.reset()
    
    def predict_step(self, batch, batch_nb):
        self.eval()
        model_input = {k: v.squeeze(0) for k, v in batch.items()}
        with torch.no_grad():
            _, logits = self.forward(model_input, pred=True)
            probs = F.softmax(logits, dim=1)
            _, total_preds = probs.max(dim = 1)

        return total_preds

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
    
    def freeze_bert_parameters(self, model):
        print("freeze_bert_para")
        for name, param in model.model.named_parameters():
            param.requires_grad = False
            # print('name', name)
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
        return model

    def draw_label(self, weights, labels):
        print("TSNE: fitting start...")
        tsne = TSNE(n_components=2, metric='cosine', random_state=0, n_jobs=4)
        embedding = tsne.fit_transform(weights)

        df = pd.DataFrame(embedding, columns=['x', 'y'])  # x, y 값을 DataFrame으로 변환
        df['label'] = labels  # 라벨을 DataFrame에 추가
        df.to_csv(self.tsne_path + '.csv', index=False)
        print("csv!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, normalize=False, title=None):
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 10))
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