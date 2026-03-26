"""
src/inference/preprocess.py

Preprocessing utilities for webcam frames before model inference.
"""

import cv2
import torch
from torchvision import transforms


def build_transform():
    """
    Build the same preprocessing pipeline used during training.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def preprocess_frame(frame):
    """
    Convert OpenCV BGR frame to RGB and transform into a model input tensor.
    """
    transform = build_transform()

    # OpenCV reads in BGR; most deep learning image pipelines expect RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    tensor = transform(rgb_frame)
    tensor = tensor.unsqueeze(0)  # Add batch dimension: [1, C, H, W]

    return tensor
