from .base import base_infer_pipeline, base_train_pipeline


def get_train_pipeline(cfg):
    match cfg.pipeline:
        case "base":
            return base_train_pipeline
        case _:
            raise ValueError(f"Invalid Pipeline Name: {cfg.pipeline}")


def get_infer_pipeline(cfg):
    match cfg.pipeline:
        case "base":
            return base_infer_pipeline
        case _:
            raise ValueError(f"Invalid Pipeline Name: {cfg.pipeline}")
