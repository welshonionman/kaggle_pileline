import timm
import torch.nn as nn

from src.constants import IS_KAGGLE_NOTEBOOK


class Base_Model(nn.Module):
    def __init__(self, encoder_name, num_classes=1, pretrained=not IS_KAGGLE_NOTEBOOK):
        super(Base_Model, self).__init__()
        self.model = timm.create_model(encoder_name, pretrained=pretrained)

    def forward(self, images):
        output = self.model(images)
        return output
