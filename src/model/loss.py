import torch
import torch.nn as nn


def get_lossfn(cfg, labels=None):
    match cfg.lossfn:
        case "BCEWithLogitsLoss":
            w = cfg.loss_weight
            weight = 1 / (w + 1)
            class_weights = torch.tensor([weight, 1 - weight]).cuda()
            weights = labels * class_weights[1] + (1 - labels) * class_weights[0]
            return nn.BCEWithLogitsLoss(weight=weights)
        case _:
            raise ValueError(f"Invalid Loss Function: {cfg.lossfn}")
