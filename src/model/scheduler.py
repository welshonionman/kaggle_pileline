from torch.optim import lr_scheduler


def get_scheduler(optimizer, cfg):
    match cfg.scheduler:
        case "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
            )

        case _:
            raise ValueError(f"Invalid Scheduler: {cfg.scheduler}")
    return scheduler
