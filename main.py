import os
import whisper
import animation_lib
from animation_lib import AnimationDirector
import spacy

# Load the language model
# If missing: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

AUDIO_DIR = "INPUT_AUDIO"
POSE_DIR = "Pose"

VALID_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.ogg', '.opus', '.m4a', '.aac']

def gen_pose(text):
    """
    Transforms text into a sequence of glosses.
    Filters auxiliaries (AUX) and splits proper nouns (PROPN) into letters.
    """
    doc = nlp(text)
    gloss_list = []

    # Set of available files for quick lookup
    available_poses = {
        os.path.splitext(f)[0].upper()
        for f in os.listdir(POSE_DIR)
    }

    for token in doc:
        # 1. Skip punctuation and whitespace
        if token.is_punct or token.is_space:
            continue

        # 2. AUXILIARY FILTER: Skip verbs like 'be', 'do', 'can'
        if token.pos_ == "AUX":
            continue

        # 3. PROPER NOUNS: Perform letter-by-letter fingerspelling
        if token.pos_ == "PROPN":
            for letter in token.text:
                char_upper = letter.upper()
                if char_upper in available_poses:
                    gloss_list.append(char_upper)
                else:
                    print(f"Letter {char_upper} not found for name {token.text}")
        else:
            # 4. STANDARD WORDS: Use the word's lemma
            gloss = token.lemma_.upper()
            if gloss in available_poses:
                gloss_list.append(gloss)
            else:
                print(f"Pose not available for: {gloss}")

    if not gloss_list:
        raise ValueError("No valid gloss generated from the provided text.")

    return gloss_list

def gen_vid(sequence_pose, output_filename):
    """Renders the video from the pose sequence."""
    director = AnimationDirector(
        poses_dir=POSE_DIR,
        output_file=output_filename
    )
    print(f"Rendering video: {output_filename}...")
    director.render_sequence(sequence_pose)

def audio_to_text(file_path):
    """Transcribes audio to text using Whisper on CPU with FP32 precision."""
    # Force model loading on CPU
    model = whisper.load_model("base", device="cpu")

    print(f"Processing audio: {file_path}")
    # fp16=False avoids warnings and improves stability on CPU
    result = model.transcribe(file_path, fp16=False)

    print("Detected text:", result["text"])
    return result["text"]

def get_audio_files(path):
    """Retrieves the list of valid audio files."""
    if not os.path.isdir(path):
        return []
    return [
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in VALID_AUDIO_FORMATS
    ]

def main():
    print("\n--- Sign Language Animation Pipeline ---")
    menu_choice = input("1. Audio\n2. Text\n--> choice: ")

    if not menu_choice.isdigit() or int(menu_choice) not in (1, 2):
        print("Invalid choice.")
        return

    text_to_process = ""
    output_filename = ""

    if int(menu_choice) == 1:
        # AUDIO MODE
        audio_files = get_audio_files(AUDIO_DIR)
        if not audio_files:
            print(f"No audio files found in folder {AUDIO_DIR}.")
            return

        for i, f in enumerate(audio_files, start=1):
            print(f"{i}. {f}")

        choice = input("\nSelect file (number): ")
        if not choice.isdigit() or not (1 <= int(choice) <= len(audio_files)):
            print("Invalid choice.")
            return

        selected_audio = os.path.join(AUDIO_DIR, audio_files[int(choice) - 1])
        output_filename = f"video_{os.path.splitext(os.path.basename(selected_audio))[0]}.mp4"

        # Get text via Whisper
        text_to_process = audio_to_text(selected_audio)

    else:
        # MANUAL TEXT MODE
        text_to_process = input("\nEnter the sentence to animate (natural language): ")
        if not text_to_process.strip():
            print("Error: Empty text.")
            return
        output_filename = "video_from_text.mp4"

    # Common phase: gloss conversion and video rendering
    try:
        pose_sequence = gen_pose(text_to_process)
        gen_vid(pose_sequence, output_filename)
        print("\nProcess completed successfully.")
    except Exception as e:
        print(f"\nError during processing: {e}")

if __name__ == "__main__":
    main()
