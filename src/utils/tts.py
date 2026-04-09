"""
src/utils/tts.py

Text-to-speech helpers.
"""
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

def init_tts():
    if pyttsx3 is None:
        return None

    engine = pyttsx3.init()
    return engine

def speak_text(engine, text: str):
    if engine is None:
        print("TTS engine unavailable. Install pyttsx3 to enable speech.")
        return

    if not text.strip():
        return

    engine.say(text)
    engine.runAndWait()