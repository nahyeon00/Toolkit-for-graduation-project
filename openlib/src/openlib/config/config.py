from __future__ import annotations

# import time
# from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


# def _get_now_str(timestamp: float) -> str:
#     """Standard format for datetimes is defined here."""
#     return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")


# modified: seongmin
def get_configurable_parameters(args):
    config = OmegaConf.load(args.config)

    # keep track of the original config file because it will be modified
    config_original: DictConfig = config.copy()

    if args.model is not None:
        config.model = OmegaConf.load(args.model)

    if args.data is not None:
        config.data = OmegaConf.load(args.data)

    if args.trainer is not None:
        config.trainer = OmegaConf.load(args.trainer)

    if args.seed is not None:
        config.seed = args.seed

    # Project Configs
    project_path = Path(
        "./results"
    )  # Path(config.project.path) / config.model.name / config.dataset.name
    (project_path / "weights").mkdir(parents=True, exist_ok=True)

    # write the original config for eventual debug (modified config at the end of the function)
    (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_original))
    # config.project.path = str(project_path)

    config.trainer.default_root_dir = str(project_path)

    # if weight_file:
    #     config.trainer.resume_from_checkpoint = weight_file

    (project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))

    return config


# Original:
# def get_configurable_parameters(
#     model_name: str | None = None,
#     config_path: Path | str | None = None,
#     weight_file: str | None = None,
# ) -> DictConfig | ListConfig:
#     if config_path is None:
#         raise ValueError(
#             "Model config path cannot be None! "
#             "Please provide a model name or path to a config file!"
#         )

#     config = OmegaConf.load(config_path)

#     # keep track of the original config file because it will be modified
#     config_original: DictConfig = config.copy()

#     # Project Configs
#     project_path = Path(
#         "./results"
#     )  # Path(config.project.path) / config.model.name / config.dataset.name
#     (project_path / "weights").mkdir(parents=True, exist_ok=True)

#     # write the original config for eventual debug (modified config at the end of the function)
#     (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_original))
#     # config.project.path = str(project_path)

#     config.trainer.default_root_dir = str(project_path)

#     # if weight_file:
#     #     config.trainer.resume_from_checkpoint = weight_file

#     (project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))

#     return config
