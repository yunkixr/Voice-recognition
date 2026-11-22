
import csv
import os

from src.audio_processing import (
    read_audio,
    normalize_audio,
    divide_into_frames
)

from src.features import extract_features


def process_speaker(file_path, speaker_name):

    print(f"\nProcessing {speaker_name}...")
    print("-" * 35)

    # Read WAV file
    samples, sample_rate = read_audio(file_path)

    print("Sample Rate:", sample_rate, "Hz")
    print("Total Samples:", len(samples))
    print("Duration:", round(len(samples) / sample_rate, 2), "seconds")

    # Normalize audio
    normalized = normalize_audio(samples)

    # Divide audio into frames
    frames = divide_into_frames(
        normalized,
        sample_rate,
        frame_duration=0.03,
        overlap=0.5
    )

    print("Total Frames:", len(frames))

    # Extract RMS, ZCR and Pitch
    features = extract_features(frames, sample_rate)

    # Add speaker label
    for item in features:
        item["speaker"] = speaker_name

    return features


def main():

    recordings = {
        "Voice_A": "src/voice_A/laki.wav",
        "Voice_B": "src/voice_B/Perempuan.wav"
    }

    all_data = []

    for speaker, file_path in recordings.items():

        if not os.path.exists(file_path):
            print(f"ERROR: Cannot find {file_path}")
            return

        try:
            features = process_speaker(file_path, speaker)
            all_data.extend(features)

        except Exception as error:
            print(f"ERROR processing {speaker}: {error}")
            return

    # Save extracted features
    output_file = "src/audio_features.csv"

    columns = [
        "speaker",
        "frame",
        "rms",
        "zcr",
        "pitch"
    ]

    with open(output_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(all_data)

    print("\n" + "=" * 40)
    print("AUDIO PROCESSING COMPLETED!")
    print("=" * 40)
    print("Total feature rows:", len(all_data))
    print("Saved dataset:", output_file)


if __name__ == "__main__":
    main()
