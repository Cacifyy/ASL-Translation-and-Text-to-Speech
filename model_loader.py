"""
src/models/model_loader.py

Load trained model weights for inference.
"""

import torch
from torchvision import models


def load_model(model_path: str, num_classes: int = 26, device: str = None):
    """
    Load the trained model and set it to evaluation mode.

    Parameters
    ----------
    model_path : str
        Path to the saved model weights.
    num_classes : int
        Number of output classes in the classifier.
    device : str
        Device to load model onto ("cpu" or "cuda").

    Returns
    -------
    model : torch.nn.Module
        Loaded model ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example: ResNet-18 backbone
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model
