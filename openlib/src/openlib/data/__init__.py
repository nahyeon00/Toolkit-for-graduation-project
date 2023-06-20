from __future__ import annotations

from omegaconf import DictConfig, ListConfig
from .data_module import OICDataModule


def get_datamodule(config: DictConfig | ListConfig) -> OICDataModule:
    datamodule = OICDataModule(
        data_path=config.data.data_path,
        dataset=config.data.dataset,
        # preprocessed_dir=config.data.preprocessed_dir,
        model_name_or_path=config.data.model_name_or_path,
        known_cls_ratio=config.data.known_cls_ratio,
        labeled_ratio=config.data.labeled_ratio, 
        # include_unknown=config.data.include_unknown, 
        max_seq_len=config.data.max_seq_len, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers,
        k_1=config.data.k_1,
    )

    return datamodule

