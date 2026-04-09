"""
demo.py

Run a live ASL recognition demo using webcam input, MediaPipe hand detection,
model inference, and optional text-to-speech output.

Controls:
SPACE: capture the current frame and predict
Q: quit
"""

import cv2
import mediapipe as mp

from src.utils.camera import open_camera, read_frame, release_camera
from src.utils.tts import init_tts, speak_text
from model_loader import load_model
from src.inference.predict import predict_from_frame


_mp_hands = mp.solutions.hands
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles


def extract_hand_crop(frame, hand_landmarks, padding: float = 0.2):
    """
    Crop the frame to a bounding box around the detected hand landmarks.
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


def draw_result(frame, bbox, label, confidence):
    """Draw bounding box and prediction label around the detected hand."""
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)
    cv2.putText(
        frame, f"{label} ({confidence:.2f})",
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA
    )


def main():
    camera_index = 0
    model_path = "models/resnet18_finetune_aug/best_model.pt"
    speak_predictions = True

    # Initialize best model, TTS, camera, and MediaPipe
    print("Loading model...")
    model = load_model(model_path)

    print("Initializing TTS...")
    tts_engine = init_tts() if speak_predictions else None

    print("Opening camera...")
    cap = open_camera(camera_index)

    print("Ready.  SPACE = capture & predict   Q = quit")

    hands = _mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    # Last prediction stays on screen until the next capture
    last_label = None
    last_confidence = 0.0
    last_bbox = None

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                print("Warning: could not read frame from webcam.")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_landmarks = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                _mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    _mp_hands.HAND_CONNECTIONS,
                    _mp_drawing_styles.get_default_hand_landmarks_style(),
                    _mp_drawing_styles.get_default_hand_connections_style(),
                )
            else:
                cv2.putText(
                    frame, "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA
                )

            if last_label is not None and last_bbox is not None:
                draw_result(frame, last_bbox, last_label, last_confidence)

            h = frame.shape[0]
            cv2.putText(
                frame, "SPACE: predict   Q: quit",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA
            )

            cv2.imshow("ASL Recognition Demo", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if hand_landmarks is None:
                    print("No hand detected — move your hand into frame and try again.")
                    continue

                hand_crop, bbox = extract_hand_crop(frame, hand_landmarks, padding=0.2)
                if hand_crop is None:
                    print("Could not crop hand region.")
                    continue

                label, confidence = predict_from_frame(model, hand_crop)
                last_label = label
                last_confidence = confidence
                last_bbox = bbox
                print(f"Predicted: {label}  ({confidence:.2f})")

                if speak_predictions:
                    speak_text(tts_engine, label)

    finally:
        hands.close()
        release_camera(cap)
        cv2.destroyAllWindows()
        print("Demo closed cleanly.")


if __name__ == "__main__":
    main()
