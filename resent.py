import torch 
from torchvision import models 
from torch import nn


class KneeResNet(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True, pretrained: bool = True):
        super().__init__()

        # Load pretrained backbone
        self.backbone =  models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        if freeze_backbone:
            self.backbone.requires_grad_(False)

        in_features = self.backbone.fc.in_features

        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)
