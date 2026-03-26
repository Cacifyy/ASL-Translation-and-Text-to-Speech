"""
src/inference/predict.py

Prediction utilities for live webcam frames.
"""

import torch

from src.inference.preprocess import preprocess_frame
from src.inference.labels import CLASS_NAMES


def predict_from_frame(model, frame, device: str = None):
    """
    Run inference on a single OpenCV frame.

    Returns
    -------
    predicted_label : str
        Predicted ASL class name.
    confidence : float
        Softmax confidence for the predicted class.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = preprocess_frame(frame).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    predicted_label = CLASS_NAMES[pred_idx]
    return predicted_label, confidence
