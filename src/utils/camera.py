"""
src/webcam/camera.py

Webcam helpers for the live demo.
"""
import cv2

def open_camera(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def release_camera(cap):
    if cap is not None:
        cap.release()