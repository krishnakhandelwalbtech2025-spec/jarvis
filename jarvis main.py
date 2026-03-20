"""
JARVIS Main Launcher
Usage:
    python main.py              → CLI text mode
    python main.py --voice      → Voice mode (mic + TTS)
    python main.py --voice --el → Voice mode with ElevenLabs TTS
"""

import argparse
import sys
import time
import re

from jarvis_core import JARVIS
from jarvis_voice import Speaker, Listener, WakeWord


# ─────────────────────────────────────────────
#  CONFIG — Edit these
# ─────────────────────────────────────────────
ELEVENLABS_API_KEY = ""        # Optional: ElevenLabs key for premium voice
PICOVOICE_ACCESS_KEY = ""      # Optional: Picovoice key for wake word
WAKE_WORD = "jarvis"
VOICE_LISTEN_SECONDS = 6       # How long to record after wake word


# ─────────────────────────────────────────────
#  RICH PRINT HELPERS
# ─────────────────────────────────────────────
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def print_jarvis(text: str):
    """Animated character-by-character JARVIS output."""
    print(f"\n{CYAN}{BOLD}  JARVIS:{RESET} ", end="", flush=True)
    for char in text:
        print(f"{CYAN}{char}{RESET}", end="", flush=True)
        time.sleep(0.012)
    print("\n")


def print_user(text: str):
    print(f"\n{YELLOW}  You:{RESET} {text}\n")


def print_status(msg: str):
    print(f"{DIM}  [{msg}]{RESET}")


# ─────────────────────────────────────────────
#  TEXT MODE (CLI)
# ─────────────────────────────────────────────
def run_text_mode(jarvis: JARVIS):
    print(f"\n{GREEN}  ● Text mode active — type your message{RESET}\n")
    print(f"{DIM}  Commands: 'exit', 'clear', 'history', 'remember: <fact>'{RESET}\n")

    while True:
        try:
            user_input = input(f"{YELLOW}  You → {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "shutdown"):
            print_jarvis("Shutting down all systems. Goodbye, Sir.")
            break

        response = jarvis.respond(user_input)
        print_jarvis(response)


# ─────────────────────────────────────────────
#  VOICE MODE
# ─────────────────────────────────────────────
def run_voice_mode(jarvis: JARVIS, use_elevenlabs: bool = False):
    print(f"\n{GREEN}  ● Voice mode active{RESET}\n")

    speaker = Speaker(
        use_elevenlabs=use_elevenlabs,
        el_api_key=ELEVENLABS_API_KEY
    )
    listener = Listener(use_whisper=True, model_size="base")
    wakeword = WakeWord(
        access_key=PICOVOICE_ACCESS_KEY,
        keyword=WAKE_WORD
    )

    greeting = "JARVIS online. Awaiting your command, Sir."
    print_jarvis(greeting)
    speaker.say(greeting)

    while True:
        try:
            # Wait for wake word (or skip if no Porcupine key)
            if PICOVOICE_ACCESS_KEY:
                print_status(f"Say '{WAKE_WORD}' to activate...")
                wakeword.wait_for_wake_word()
                print_status("Wake word detected!")
                speaker.say("Yes, Sir?")
            else:
                input(f"\n{DIM}  [Press Enter to speak]{RESET}")

            # Listen to user
            text = listener.listen(duration=VOICE_LISTEN_SECONDS)
            if not text:
                speaker.say("I didn't catch that, Sir.")
                continue

            print_user(text)

            if any(word in text.lower() for word in ["exit", "shutdown", "goodbye"]):
                msg = "Shutting down. Goodbye, Sir."
                print_jarvis(msg)
                speaker.say(msg, blocking=True)
                break

            # Get JARVIS response
            print_status("Processing...")
            response = jarvis.respond(text)
            print_jarvis(response)
            speaker.say(response)

        except KeyboardInterrupt:
            print("\n")
            speaker.say("Emergency shutdown initiated.", blocking=True)
            break

    wakeword.cleanup()
    speaker.stop()


# ─────────────────────────────────────────────
#  HYBRID MODE (type but JARVIS speaks back)
# ─────────────────────────────────────────────
def run_hybrid_mode(jarvis: JARVIS, use_elevenlabs: bool = False):
    print(f"\n{GREEN}  ● Hybrid mode: type input, JARVIS speaks output{RESET}\n")

    speaker = Speaker(
        use_elevenlabs=use_elevenlabs,
        el_api_key=ELEVENLABS_API_KEY
    )
    greeting = "JARVIS online. I'll speak my responses, Sir."
    print_jarvis(greeting)
    speaker.say(greeting)

    while True:
        try:
            user_input = input(f"{YELLOW}  You → {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            msg = "Shutting down. Goodbye."
            print_jarvis(msg)
            speaker.say(msg, blocking=True)
            break

        response = jarvis.respond(user_input)
        print_jarvis(response)
        speaker.say(response)

    speaker.stop()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="JARVIS AI Assistant")
    parser.add_argument("--voice", action="store_true", help="Enable voice I/O")
    parser.add_argument("--hybrid", action="store_true", help="Type input, hear output")
    parser.add_argument("--el", action="store_true", help="Use ElevenLabs TTS")
    args = parser.parse_args()

    jarvis = JARVIS()

    if args.voice:
        run_voice_mode(jarvis, use_elevenlabs=args.el)
    elif args.hybrid:
        run_hybrid_mode(jarvis, use_elevenlabs=args.el)
    else:
        run_text_mode(jarvis)


if __name__ == "__main__":
    main()