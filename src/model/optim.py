from torch import optim


def get_optimizer(model, cfg):
    match cfg.optimizer:
        case "Adam":
            optimizer = optim.Adam(
                model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        case "AdamW":
            optimizer = optim.AdamW(
                model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        case _:
            raise ValueError(f"Invalid Optimizer: {cfg.optimizer}")
    return optimizer
