
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.audio_processing import (
    read_audio,
    normalize_audio,
    divide_into_frames
)
from src.features import extract_features
from src.model import predict, standardize, predict_label


TIMELINE = [
    (0, 3, "B"),
    (3, 9, "A"),
    (9, 12, "B"),
    (12, 18, "A"),
    (18, 21, "B"),
    (21, 27, "A"),
    (27, 33, "B"),
    (33, 36, "A"),
    (36, 39, "B"),
    (39, 42, "A")
]


def get_actual_speaker(timestamp):
    for start, end, speaker in TIMELINE:
        if start <= timestamp < end:
            return speaker

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        default="src/conversation/conversation.wav"
    )
    parser.add_argument(
        "--model",
        default="speaker_model.json"
    )
    parser.add_argument(
        "--output",
        default="output/conversation_predictions.csv"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with model_path.open("r", encoding="utf-8") as file:
        model_data = json.load(file)

    samples, sample_rate = read_audio(str(audio_path))

    if sample_rate != 48000:
        raise ValueError(
            f"Expected 48000 Hz, got {sample_rate} Hz."
        )

    duration = len(samples) / sample_rate

    if duration < 42:
        raise ValueError(
            f"Recording is only {duration:.2f} seconds long. "
            "The provided timeline requires at least 42 seconds."
        )

    normalized = normalize_audio(samples)

    frames = divide_into_frames(
        normalized,
        sample_rate,
        frame_duration=0.03,
        overlap=0.5
    )

    extracted = extract_features(frames, sample_rate)

    frame_size = int(sample_rate * 0.03)
    hop_size = int(frame_size * 0.5)

    feature_map = {
        "pitch_hz": "pitch",
        "zcr": "zcr",
        "rms": "rms"
    }

    X = []
    records = []

    for index, item in enumerate(extracted):
        timestamp = index * hop_size / sample_rate
        midpoint = timestamp + 0.015

        actual = get_actual_speaker(midpoint)

        if actual is None or item["rms"] <= 0:
            continue

        row = [
            item[feature_map[name]]
            for name in model_data["feature_names"]
        ]

        X.append(row)

        records.append({
            "clip_id": f"Conversation_{index + 1}",
            "timestamp_seconds": round(midpoint, 4),
            "actual": actual
        })

    if not X:
        raise ValueError("No usable audio frames were found.")

    X = np.asarray(X, dtype=float)

    mean = np.asarray(model_data["mean"], dtype=float)
    std = np.asarray(model_data["std"], dtype=float)
    weights = np.asarray(model_data["weights"], dtype=float)
    bias = float(model_data["bias"])
    threshold = float(model_data["threshold"])

    if np.any(std == 0):
        raise ValueError("Model contains a zero standard deviation.")

    X_scaled = standardize(X, mean, std)

    scores = predict(X_scaled, weights, bias)
    labels = predict_label(scores, threshold)

    label_map = model_data["labels"]

    for record, score, label in zip(records, scores, labels):
        record["predicted"] = label_map[str(int(label))]
        record["score"] = float(score)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "clip_id",
                "timestamp_seconds",
                "actual",
                "predicted",
                "score"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nCOMBINED CONVERSATION PREDICTION")
    print("-" * 40)
    print("Recording duration:", round(duration, 2), "seconds")
    print("Total frames:", len(frames))
    print("Usable frames:", len(records))
    print("Predictions saved to:", output_path)


if __name__ == "__main__":
    main()
