import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
)


class TransformerFeatureExtractor(nn.Module):
    """Extract feature using Transformer

    Examples:
        With custom models:

            >>> from src.openlib.models.base.feature_extractor import TransformerFeatureExtractor
            >>> feature_extractor = TransformerFeatureExtractor(
                    model_name_or_path="path.to.checkpoints",
                )
            >>> features = feature_extractor(input)
    """

    def __init__(
        self,
        model_name_or_path: str,
        dropout_prob: float = 0.5,
    ):
        super().__init__()

        # Use pretrained language model
        
        self.model, self.dense = self.initialize_feature_extractor(model_name_or_path)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, batch):
        outputs = self.model(**batch)

        mean_pooling = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dense(mean_pooling)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        return pooled_output

    def initialize_feature_extractor(self, model_name_or_path: str):
        if model_name_or_path.endswith("ckpt"):
            pretrained = torch.load(model_name_or_path)
            model_name_or_path = pretrained["hyper_parameters"]["model_name_or_path"]
            state_dict = pretrained["state_dict"]
            for key in list(state_dict.keys()):
                state_dict[key.replace("model.model.", "")] = state_dict.pop(key)
            # print("dense", state_dict.keys())

            dense_w = state_dict.pop("model.dense.weight")
            dense_b = state_dict.pop("model.dense.bias")
            
            model = AutoModel.from_pretrained(model_name_or_path, state_dict=state_dict)

            dense = nn.Linear(
            model.config.hidden_size, model.config.hidden_size
            )
            
            dense.weight.data = dense_w
            dense.bias.data = dense_b

        else:
            config = AutoConfig.from_pretrained(model_name_or_path)
            model = AutoModel.from_pretrained(model_name_or_path, config=config)
            dense = nn.Linear(
            model.config.hidden_size, model.config.hidden_size
            )

        return model, dense
