"""
demo.py

Run a live ASL recognition demo using webcam input, MediaPipe hand detection,
model inference, and optional text-to-speech output.

Hand landmarks are drawn on screen so you can see exactly what the model
is looking at. Inference runs on the cropped hand region, not the full frame.
"""

import cv2
import time
import numpy as np
import mediapipe as mp

from src.utils.camera import open_camera, read_frame, release_camera
from src.utils.tts import init_tts, speak_text
from model_loader import load_model
from predict import predict_from_frame


# MediaPipe hand solution
_mp_hands = mp.solutions.hands
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


def extract_hand_crop(frame, hand_landmarks, padding: float = 0.2):
    """
    Crop the frame to a bounding box around the detected hand landmarks,
    with a percentage-based padding on each side.

    Returns the cropped BGR image, or None if the crop is invalid.
    """
    h, w = frame.shape[:2]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min = int(max(0, (min(xs) - padding) * w))
    x_max = int(min(w, (max(xs) + padding) * w))
    y_min = int(max(0, (min(ys) - padding) * h))
    y_max = int(min(h, (max(ys) + padding) * h))

    if x_max <= x_min or y_max <= y_min:
        return None, None

    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def draw_hand_box(frame, bbox, label, confidence):
    """Draw bounding box and prediction label around the detected hand."""
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)
    text = f"{label} ({confidence:.2f})"
    cv2.putText(
        frame, text,
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA
    )


def main():
    # -----------------------------
    # Config
    # -----------------------------
    camera_index = 0
    model_path = "models/resnet18_finetune_aug/best_model.pt"
    speak_predictions = True

    speak_cooldown_seconds = 2.0
    last_spoken_label = None
    last_spoken_time = 0.0

    # -----------------------------
    # Initialize model + TTS + camera + MediaPipe
    # -----------------------------
    print("Loading model...")
    model = load_model(model_path)

    print("Initializing TTS...")
    tts_engine = init_tts() if speak_predictions else None

    print("Opening camera...")
    cap = open_camera(camera_index)

    print("Starting demo. Press 'q' to quit.")
    print("  - Hand landmarks are drawn in real time.")
    print("  - Inference runs on the cropped hand region.")
    print("  - If no hand is detected, the label shows 'no hand'.")

    hands = _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                print("Warning: could not read frame from webcam.")
                continue

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            predicted_label = "no hand"
            confidence = 0.0

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw the 21-point skeleton on the frame
                _mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    _mp_hands.HAND_CONNECTIONS,
                    _mp_drawing_styles.get_default_hand_landmarks_style(),
                    _mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Crop to the hand and run inference on that region only
                hand_crop, bbox = extract_hand_crop(frame, hand_landmarks, padding=0.2)
                if hand_crop is not None:
                    predicted_label, confidence = predict_from_frame(model, hand_crop)
                    draw_hand_box(frame, bbox, predicted_label, confidence)
            else:
                # No hand found — show a message so the user knows
                cv2.putText(
                    frame, "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )

            cv2.imshow("ASL Recognition Demo", frame)

            # Speak the prediction if new and cooldown has passed
            current_time = time.time()
            if (
                speak_predictions
                and predicted_label not in ("no hand",)
                and predicted_label != last_spoken_label
                and (current_time - last_spoken_time) >= speak_cooldown_seconds
            ):
                speak_text(tts_engine, predicted_label)
                last_spoken_label = predicted_label
                last_spoken_time = current_time

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        hands.close()
        release_camera(cap)
        cv2.destroyAllWindows()
        print("Demo closed cleanly.")


if __name__ == "__main__":
    main()
