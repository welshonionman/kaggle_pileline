from .base import infer_base_preprocess, train_base_preprocess


def get_train_preprocess(cfg):
    match cfg.preprocess:
        case "base":
            return train_base_preprocess
        case _:
            raise ValueError(f"Invalid Preprocess Name: {cfg.preprocess}")


def get_infer_preprocess(cfg):
    match cfg.preprocess:
        case "base":
            return infer_base_preprocess
        case _:
            raise ValueError(f"Invalid Preprocess Name: {cfg.preprocess}")
