from torch.utils.data import DataLoader

from src.constants import TEST_HDF

from .base import (
    Base_Test_Dataset,
    Base_Train_Dataset,
    Base_Valid_Dataset,
)



def get_train_dataloader(df, fold, cfg):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    match cfg.dataset:
        case "base":
            train_dataset = Base_Train_Dataset(df_train, cfg)
            valid_dataset = Base_Valid_Dataset(df_valid, cfg)
        case _:
            raise ValueError(f"Invalid Dataset Name: {cfg.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader


def get_test_dataloader(df, cfg):
    match cfg.dataset:
        case "base":
            test_dataset = Base_Test_Dataset(df, TEST_HDF, cfg)
        case _:
            raise ValueError(f"Invalid Dataset Name: {cfg.pipeline}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    return test_loader
