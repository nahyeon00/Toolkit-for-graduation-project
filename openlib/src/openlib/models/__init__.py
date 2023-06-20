from __future__ import annotations

import os
from importlib import import_module

from omegaconf import DictConfig, ListConfig
from torch import load


def get_model(config: DictConfig | ListConfig):
    module_path, model_name = config.model.class_path.rsplit(".", 1)
    module = import_module(f"src.openlib.models.{module_path}")
    model = getattr(module, f"{model_name}")(**config.model.init_args)

    # if "init_weights" in config.keys() and config.init_weights:
    #     model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))["state_dict"], strict=False)

    return model
