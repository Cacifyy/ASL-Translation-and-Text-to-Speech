# ASL-Translation-and-Text-to-Speech

Real-time American Sign Language recognition using a fine-tuned ResNet-18 model, with live webcam inference and text-to-speech output.

## Setup

**Requirements:** Python 3.11, a webcam

```bash
# Clone the repo
git clone https://github.com/Cacifyy/ASL-Translation-and-Text-to-Speech.git
cd ASL-Translation-and-Text-to-Speech

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Getting the model

The trained model (`best_model.pt`) is not tracked in git. Place it at:

```
models/resnet18_finetune_aug/best_model.pt
```

It can be obtained by running `asl-experiments-kaggle.ipynb` on Kaggle, which trains and saves the model.

## Running the demo

```bash
source .venv/bin/activate
python demo.py
```

The webcam window will open and display the predicted ASL letter in real time. Predictions are spoken aloud via text-to-speech. Press `q` to quit.

## Project structure

```
demo.py               # Entry point — webcam loop, inference, TTS
model_loader.py       # Loads best_model.pt for inference
predict.py            # Runs a single frame through the model
preprocess.py         # Resizes and normalizes frames to match training
labels.py             # Class name list (A–Z, del, nothing, space)
src/utils/
  camera.py           # Webcam open/read/release helpers
  tts.py              # pyttsx3 text-to-speech helpers
models/               # Saved model weights (not tracked in git)
outputs/              # Training metrics and evaluation results (not tracked)
asl-experiments-kaggle.ipynb  # Training notebook (runs on Kaggle)
```