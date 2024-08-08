import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Base_Train_Dataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df.reset_index()
        self.file_names = self.df["file_path"].values
        self.targets = self.df["target"].values
        self.transforms = cfg.train_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path, -1)
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "target": target}


class Base_Valid_Dataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df.reset_index()
        self.file_names = self.df["file_path"].values
        self.targets = self.df["target"].values
        self.transforms = cfg.valid_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path, -1)
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "target": target}


class Base_Test_Dataset(Dataset):
    def __init__(self, df, file_hdf, cfg):
        self.df = df.reset_index()
        self.file_names = self.df["file_path"].values
        self.targets = self.df["target"].values
        self.transforms = cfg.valid_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path, -1)
        target = torch.tensor(self.targets[index])

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {"image": img, "target": target}
