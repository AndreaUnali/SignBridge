import subprocess
import sys
import os
import shutil

def check_ffmpeg():
    """Checks if FFmpeg is installed on macOS."""
    if shutil.which("ffmpeg") is None:
        print("\n[ERROR] FFmpeg not found!")
        print("On Mac, the fastest way to install it is via Homebrew:")
        print("Run in the terminal: brew install ffmpeg")
        return False
    return True

def run_command(command):
    """Runs a command using the current interpreter."""
    try:

        subprocess.check_call([sys.executable, "-m"] + command)
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        sys.exit(1)

def main():
    print("--- Project Setup (macOS) ---")

    if not check_ffmpeg():
        sys.exit(1)

    print("\n1. Upgrading pip...")
    run_command(["pip", "install", "--upgrade", "pip"])

    print("\n2. Installing libraries...")
    run_command(["pip", "install", "-r", "requirements.txt"])

    print("\n3. Downloading spaCy language model...")

    run_command(["spacy", "download", "en_core_web_sm"])

    print("\n--- Installation Complete ---")

if __name__ == "__main__":
    main()
