"""
JARVIS Voice Module
Handles: Text-to-Speech (TTS) + Speech-to-Text (STT) + Wake Word
"""

import threading
import queue
import time
from typing import Optional


# ─────────────────────────────────────────────
#  TTS ENGINE
# ─────────────────────────────────────────────
class Speaker:
    """
    Text-to-Speech using pyttsx3 (offline) or ElevenLabs (premium).
    Install: pip install pyttsx3
    Optional: pip install elevenlabs
    """
    def __init__(self, use_elevenlabs: bool = False, el_api_key: str = ""):
        self.use_elevenlabs = use_elevenlabs and bool(el_api_key)
        self.el_api_key = el_api_key
        self._engine = None
        self._speech_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._init_engine()

    def _init_engine(self):
        if self.use_elevenlabs:
            return  # ElevenLabs handles its own session
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            voices = self._engine.getProperty("voices")
            # Try to pick a male-ish voice (index varies by OS)
            if len(voices) > 1:
                self._engine.setProperty("voice", voices[1].id)
            self._engine.setProperty("rate", 172)
            self._engine.setProperty("volume", 0.95)
        except ImportError:
            print("[Voice] pyttsx3 not found — TTS disabled. pip install pyttsx3")
            self._engine = None

    def _speak_pyttsx3(self, text: str):
        if self._engine:
            self._engine.say(text)
            self._engine.runAndWait()

    def _speak_elevenlabs(self, text: str):
        try:
            from elevenlabs import generate, play, set_api_key
            set_api_key(self.el_api_key)
            audio = generate(
                text=text,
                voice="Adam",       # Or "Josh", "Sam" — pick your JARVIS voice
                model="eleven_monolingual_v1"
            )
            play(audio)
        except ImportError:
            print("[Voice] elevenlabs not installed. pip install elevenlabs")

    def _worker(self):
        """Background thread processes speech queue."""
        while True:
            text = self._speech_queue.get()
            if text is None:
                break
            if self.use_elevenlabs:
                self._speak_elevenlabs(text)
            else:
                self._speak_pyttsx3(text)
            self._speech_queue.task_done()

    def say(self, text: str, blocking: bool = False):
        """Queue text for speech. Non-blocking by default."""
        self._speech_queue.put(text)
        if blocking:
            self._speech_queue.join()

    def stop(self):
        self._speech_queue.put(None)


# ─────────────────────────────────────────────
#  STT ENGINE (Microphone → Text)
# ─────────────────────────────────────────────
class Listener:
    """
    Speech-to-Text using OpenAI Whisper (local, offline).
    Install: pip install openai-whisper sounddevice soundfile numpy
    Or use Google STT: pip install SpeechRecognition pyaudio
    """
    def __init__(self, use_whisper: bool = True, model_size: str = "base"):
        self.use_whisper = use_whisper
        self.model = None
        if use_whisper:
            self._load_whisper(model_size)

    def _load_whisper(self, size: str):
        try:
            import whisper
            print(f"[Voice] Loading Whisper '{size}' model...")
            self.model = whisper.load_model(size)
            print("[Voice] Whisper ready.")
        except ImportError:
            print("[Voice] Whisper not found — pip install openai-whisper")
            self.use_whisper = False

    def listen_whisper(self, duration: int = 5) -> Optional[str]:
        """Record audio then transcribe with Whisper."""
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            import tempfile

            print("  [Listening...]")
            sample_rate = 16000
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sample_rate)
                result = self.model.transcribe(f.name)
                return result["text"].strip()
        except Exception as e:
            print(f"[Voice] Whisper error: {e}")
            return None

    def listen_google(self) -> Optional[str]:
        """Fallback: Google Speech Recognition (requires internet)."""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("  [Listening...]")
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=10, phrase_time_limit=8)
            text = r.recognize_google(audio)
            return text
        except Exception as e:
            print(f"[Voice] Google STT error: {e}")
            return None

    def listen(self, duration: int = 5) -> Optional[str]:
        if self.use_whisper and self.model:
            return self.listen_whisper(duration)
        return self.listen_google()


# ─────────────────────────────────────────────
#  WAKE WORD DETECTOR
# ─────────────────────────────────────────────
class WakeWord:
    """
    Wake word detection using Porcupine (offline, fast).
    Install: pip install pvporcupine pvrecorder
    Free tier supports "Jarvis", "Hey Siri", etc. with a free API key.
    Get key: https://console.picovoice.ai/
    """
    def __init__(self, access_key: str = "", keyword: str = "jarvis"):
        self.access_key = access_key
        self.keyword = keyword
        self.porcupine = None
        self.recorder = None
        self._active = False

        if access_key:
            self._init_porcupine()

    def _init_porcupine(self):
        try:
            import pvporcupine
            import pvrecorder

            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[self.keyword]
            )
            self.recorder = pvrecorder.PvRecorder(
                device_index=-1,
                frame_length=self.porcupine.frame_length
            )
            print(f"[Wake Word] Listening for '{self.keyword}'...")
        except ImportError:
            print("[Wake Word] pvporcupine not installed — wake word disabled.")
        except Exception as e:
            print(f"[Wake Word] Init error: {e}")

    def wait_for_wake_word(self) -> bool:
        """Blocking — returns True when wake word detected."""
        if not self.porcupine or not self.recorder:
            return True  # Fallback: always active

        self.recorder.start()
        try:
            while True:
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                if result >= 0:
                    return True
        except Exception as e:
            print(f"[Wake Word] Error: {e}")
            return True
        finally:
            self.recorder.stop()

    def cleanup(self):
        if self.recorder:
            self.recorder.delete()
        if self.porcupine:
            self.porcupine.delete()