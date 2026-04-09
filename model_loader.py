"""
src/models/model_loader.py

Load trained model weights for inference.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18ASL(nn.Module):
    """ResNet-18 wrapper matching the architecture used during training."""

    def __init__(self, num_classes: int = 29):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def load_model(model_path: str, num_classes: int = 29, device: str = None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNet18ASL(num_classes=num_classes)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model
