from .base import Base_Model


def get_model(cfg):
    match cfg.model_name:
        case "base":
            model = Base_Model(cfg.encoder_name)
        case _:
            raise ValueError(f"Invalid Model Name: {cfg.model_name}")

    return model
