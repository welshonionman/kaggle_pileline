import os

import torch

COMP_NAME = "ISIC2024"
ROOT_DIR = "/kaggle/input/isic-2024-challenge"
TRAIN_DIR = f"{ROOT_DIR}/train-image/image"
TEST_CSV = f"{ROOT_DIR}/test-metadata.csv"
SAMPLE = f"{ROOT_DIR}/sample_submission.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_KAGGLE_NOTEBOOK = any("KAGGLE" in item for item in dict(os.environ).keys())
