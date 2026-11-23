
import csv
import json
from pathlib import Path

import numpy as np

from src.config import (
    FEATURE_NAMES,
    SPEAKER_TO_LABEL,
    LABEL_TO_SPEAKER,
    LEARNING_RATE,
    MAX_EPOCHS,
    TOLERANCE
)
from src.model import fit_standardizer, standardize, predict, predict_label
from src.gradient_descent import gradient_descent


INPUT_FILE = Path("output/audio_features.csv")
MODEL_FILE = Path("speaker_model.json")
PREDICTIONS_FILE = Path("output/predictions.csv")

TEST_RATIO = 0.20
RANDOM_SEED = 42
FRAME_GAP = 2


def load_dataset(path):
    speakers = {"Voice_A": [], "Voice_B": []}

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required = {"speaker", "frame", "rms", "zcr", "pitch"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Invalid audio_features.csv columns.")

        for row in reader:
            speaker = row["speaker"]

            if speaker not in speakers:
                continue

            rms = float(row["rms"])

            if rms <= 0:
                continue

            features = [
                float(row["pitch"]),
                float(row["zcr"]),
                rms
            ]

            if not np.all(np.isfinite(features)):
                continue

            speakers[speaker].append({
                "frame": row["frame"],
                "features": features,
                "label": SPEAKER_TO_LABEL[speaker[-1]]
            })

    return speakers


def split_dataset(speakers):
    training = {"Voice_A": [], "Voice_B": []}
    testing = {"Voice_A": [], "Voice_B": []}

    for speaker, rows in speakers.items():
        split_index = int(len(rows) * (1 - TEST_RATIO))

        training[speaker] = rows[:split_index - FRAME_GAP]
        testing[speaker] = rows[split_index:]

        if not training[speaker] or not testing[speaker]:
            raise ValueError(f"Not enough usable frames for {speaker}.")

    return training, testing


def balance_training_data(training):
    count = min(len(rows) for rows in training.values())

    balanced = []

    for rows in training.values():
        indices = np.linspace(
            0,
            len(rows) - 1,
            count,
            dtype=int
        )

        balanced.extend(rows[index] for index in indices)

    return balanced


def save_model(weights, bias, mean, std, final_loss, epochs):
    model_data = {
        "feature_names": list(FEATURE_NAMES),
        "weights": weights.tolist(),
        "bias": float(bias),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": 0.5,
        "final_loss": float(final_loss),
        "epochs": int(epochs),
        "labels": LABEL_TO_SPEAKER
    }

    with MODEL_FILE.open("w", encoding="utf-8") as file:
        json.dump(model_data, file, indent=4)


def save_predictions(testing, weights, bias, mean, std):
    records = []

    for speaker, rows in testing.items():
        X = np.array(
            [row["features"] for row in rows],
            dtype=float
        )

        X_scaled = standardize(X, mean, std)
        scores = predict(X_scaled, weights, bias)
        predictions = predict_label(scores)

        for row, score, prediction in zip(rows, scores, predictions):
            records.append({
                "clip_id": f"{speaker}_{row['frame']}",
                "actual": LABEL_TO_SPEAKER[row["label"]],
                "predicted": LABEL_TO_SPEAKER[int(prediction)],
                "score": float(score)
            })

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with PREDICTIONS_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["clip_id", "actual", "predicted", "score"]
        )

        writer.writeheader()
        writer.writerows(records)

    return records


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    speakers = load_dataset(INPUT_FILE)

    print("Usable frames:")

    for speaker, rows in speakers.items():
        print(f"{speaker}: {len(rows)}")

    training, testing = split_dataset(speakers)

    balanced = balance_training_data(training)

    X_train = np.array(
        [row["features"] for row in balanced],
        dtype=float
    )

    y_train = np.array(
        [row["label"] for row in balanced],
        dtype=float
    )

    mean, std = fit_standardizer(X_train)
    std = np.where(std == 0, 1.0, std)

    X_train = standardize(X_train, mean, std)

    weights, bias, loss_history = gradient_descent(
        X_train,
        y_train,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        tolerance=TOLERANCE
    )

    if not loss_history or not np.all(np.isfinite(weights)):
        raise RuntimeError("Training failed. Check the model and features.")

    save_model(
        weights,
        bias,
        mean,
        std,
        loss_history[-1],
        len(loss_history)
    )

    records = save_predictions(
        testing,
        weights,
        bias,
        mean,
        std
    )

    correct = sum(
        row["actual"] == row["predicted"]
        for row in records
    )

    accuracy = correct / len(records) if records else 0

    print("\nTraining complete.")
    print(f"Training frames per speaker: {len(balanced) // 2}")
    print(f"Epochs: {len(loss_history)}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Test frames: {len(records)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
