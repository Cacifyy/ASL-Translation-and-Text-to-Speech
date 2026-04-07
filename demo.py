"""
demo.py

Run a live ASL recognition demo using webcam input, model inference,
and optional text-to-speech output.
"""

import cv2
import time

from src.utils.camera import open_camera, read_frame, release_camera
from src.utils.tts import init_tts, speak_text
from model_loader import load_model
from predict import predict_from_frame


def main():
    # -----------------------------
    # Config
    # -----------------------------
    camera_index = 0
    model_path = "models/resnet18_finetune_aug/best_model.pt"
    speak_predictions = True

    # To avoid repeating speech every frame, we keep track of the last
    # spoken label and only speak again after a cooldown.
    speak_cooldown_seconds = 2.0
    last_spoken_label = None
    last_spoken_time = 0.0

    # -----------------------------
    # Initialize model + TTS + camera
    # -----------------------------
    print("Loading model...")
    model = load_model(model_path)

    print("Initializing TTS...")
    tts_engine = init_tts() if speak_predictions else None

    print("Opening camera...")
    cap = open_camera(camera_index)

    print("Starting demo. Press 'q' to quit.")

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                print("Warning: could not read frame from webcam.")
                continue

            # Run prediction on current frame
            predicted_label, confidence = predict_from_frame(model, frame)

            # Draw prediction on the displayed frame
            display_text = f"{predicted_label} ({confidence:.2f})"
            cv2.putText(
                frame,
                display_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("ASL Recognition Demo", frame)

            # Speak the prediction if it is new and enough time has passed
            current_time = time.time()
            if (
                speak_predictions
                and predicted_label != last_spoken_label
                and (current_time - last_spoken_time) >= speak_cooldown_seconds
            ):
                speak_text(tts_engine, predicted_label)
                last_spoken_label = predicted_label
                last_spoken_time = current_time

            # Quit on 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        release_camera(cap)
        cv2.destroyAllWindows()
        print("Demo closed cleanly.")


if __name__ == "__main__":
    main()
