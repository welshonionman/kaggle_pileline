from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.pipeline import get_train_pipeline


class CFG:
    wandb_mode = "online"
    exp_name = Path(__file__).stem

    seed = 42
    epochs = 30
    img_size = 384
    n_fold = 5

    pipeline = "base"
    preprocess = "base"
    dataset = "base"
    model_name: str = "base"
    encoder_name: str = "tf_efficientnet_b0_ns"

    train_batch_size = 32
    valid_batch_size = 64
    learning_rate = 1e-4

    lossfn = "BCEWithLogitsLoss"
    loss_weight = 1
    sampling_factor = 20
    optimizer = "Adam"
    scheduler = "CosineAnnealingLR"

    min_lr = 1e-7
    weight_decay = 1e-6

    train_transform = A.Compose(
        [
            ToTensorV2(),
        ],
        p=1.0,
    )

    valid_transform = A.Compose(
        [
            ToTensorV2(),
        ],
        p=1.0,
    )


if __name__ == "__main__":
    pipeline = get_train_pipeline(CFG)
    pipeline(CFG)
