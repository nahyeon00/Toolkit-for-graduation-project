import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
)
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput

class BertClassificationHead(nn.Module):
    def __init__(self, config, num_labels, dropout_prob):
        super(BertClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class BertContrastiveHead(nn.Module):
    def __init__(self, config, dropout_prob):
        super(BertContrastiveHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class Base_KNNCL(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.1,
        num_classes: int = None,  # open 없는 개수
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        scheduler_type: str = "linear",
        warmup_steps: int = 0,
    ):
        super().__init__()
        # print("\nbase_knncl\n")

        # Use pretrained language model
        self.encoder_q = self.initialize_feature_extractor(model_name_or_path)
        self.encoder_k = self.initialize_feature_extractor(model_name_or_path)

        self.classifier_linear = BertClassificationHead(self.encoder_k.config, num_labels=num_classes, dropout_prob=dropout_prob)

        self.contrastive_liner_q = BertContrastiveHead(self.encoder_k.config, dropout_prob=dropout_prob)
        self.contrastive_liner_k = BertContrastiveHead(self.encoder_k.config, dropout_prob=dropout_prob)

        self.m = 0.999
        self.T = 0.5  # args.temperature  -> hyperparameter
        self.init_weights()  # Exec
        self.contrastive_rate_in_training = 0.1  # arg.contrastive_rate_in_training  -> hyperparameter

        # create the label_queue and feature_queue
        self.K = 7500  # args.quesize  -> hyperparmeter (데이터마다 다르게 함)
        self.num_classes = num_classes

        self.register_buffer("label_queue", torch.randint(0, num_classes+1, [self.K]))  # Tensor:(7500,)
        self.register_buffer("feature_queue", torch.randn(self.K, self.encoder_k.config.hidden_size))  # Tensor:(7500, 768)
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # Tensor(1,)
        self.top_k = 25  # args.top_k  -> hyperparamter
        self.update_num = 3  # args.positive_num -> hyperparamter
        self.device = next(self.encoder_q.parameters()).device
    
    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

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

    def forward(self, batch, positive_sample=None, negative_sample=None):
        if "labels" in batch.keys():  ## train
            # print("label in batch")
            labels = batch['labels']
            # print("labels", labels.shape)
            labels = labels.view(-1)

            with torch.no_grad():
                self.update_encoder_k()
                update_sample = self.reshape_dict(positive_sample)
                bert_output_p = self.encoder_k(**update_sample)
                update_keys = bert_output_p.last_hidden_state.mean(dim=1)
                # update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                update_keys = self.l2norm(update_keys)
                tmp_labels = labels.unsqueeze(-1)
                tmp_labels = tmp_labels.repeat([1, self.update_num])
                tmp_labels = tmp_labels.view(-1)
                self._dequeue_and_enqueue(update_keys, tmp_labels)
            
            model_input = {k: v for k, v in batch.items() if k != "labels"}  

            bert_output_q = self.encoder_q(**model_input)
            # q = bert_output_q[1]
            q = bert_output_q.last_hidden_state.mean(dim=1)
            liner_q = self.contrastive_liner_q(q)
            liner_q = self.l2norm(liner_q)
            logits_cls = self.classifier_linear(q)

            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.num_classes), labels)
        
            logits_con = self.select_pos_neg_sample(liner_q, labels)

            if logits_con is not None:
                labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
                loss_fct = CrossEntropyLoss()
                loss_con = loss_fct(logits_con, labels_con)

                loss = loss_con * self.contrastive_rate_in_training + \
                   loss_cls * (1 - self.contrastive_rate_in_training)
            else:
                loss = loss_cls
            loss = SequenceClassifierOutput(loss)
            return loss 
        
        else:  # valid, test # lof fit
            seq_embed = self.encoder_q(**batch)
            seq_embed = seq_embed.last_hidden_state.mean(dim=1)
            # breakpoint()

            logits_cls = self.classifier_linear(seq_embed)
            probs = torch.softmax(logits_cls, dim=1)
            return probs, seq_embed


    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def l2norm(self, x: torch.Tensor):
        norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return x

    def select_pos_neg_sample(self, liner_q, label_q):
        label_queue = self.label_queue.clone().detach()  # K
        feature_queue = self.feature_queue.clone().detach()  # K * hidden_size

        # 1. expand label_queue and feature_queue to batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1])  # batch_size * K * hidden_size

        # 2.caluate sim
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3. get index of postive and neigative 
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4.another option 
        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5.topk
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = cos_sim.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]  # self.topk = 25
        pos_sample = pos_sample_top_k
        pos_sample = pos_sample.contiguous().view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_topk = min(pos_min, self.top_k)
        neg_sample = neg_sample.repeat([1, neg_topk])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def _dequeue_and_enqueue(self, keys, label):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            batch_size = self.K - ptr
            keys = keys[: batch_size]
            label = label[: batch_size]

        # replace the keys at ptr (dequeue ans enqueue)
        self.feature_queue[ptr: ptr + batch_size, :] = keys
        self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr
