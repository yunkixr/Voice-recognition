import csv
import numpy as np

from src.config import FEATURE_NAMES, SPEAKER_TO_LABEL, LABEL_TO_SPEAKER
from src.model import predict, half_mse_loss, fit_standardizer, standardize, predict_label
from src.gradient_descent import gradient_descent


FEATURE_CSV = "output/audio_features.csv"
PREDICTIONS_CSV = "output/predictions.csv"
CSV_COLUMN = {"pitch_hz": "pitch", "zcr": "zcr", "rms": "rms"}


def load_split():
    with open(FEATURE_CSV, newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        if not {"speaker", "frame", "rms", "zcr", "pitch"} <= columns:
            raise ValueError("CSV must have speaker, frame, rms, zcr, pitch")
        rows = list(reader)

    X_train, y_train = [], []
    X_test, y_test, test_ids = [], [], []

    for recording, speaker in (("Voice_A", "A"), ("Voice_B", "B")):
        recording_rows = [row for row in rows if row["speaker"] == recording]
        cut = int(len(recording_rows) * 0.7)

        for row in recording_rows:
            if float(row["rms"]) < 0.01:
                continue

            features = [float(row[CSV_COLUMN[name]]) for name in FEATURE_NAMES]
            frame = int(row["frame"])
            label = SPEAKER_TO_LABEL[speaker]

            if frame <= cut:
                X_train.append(features)
                y_train.append(label)
            elif frame > cut + 20:
                X_test.append(features)
                y_test.append(label)
                test_ids.append(f"{speaker}_frame{frame}")

    if not X_train or not X_test:
        raise ValueError("Not enough non-silent frames for training and testing")

    return (np.array(X_train, dtype=float), np.array(y_train),
            np.array(X_test, dtype=float), np.array(y_test), test_ids)


def main():
    X_train, y_train, X_test, y_test, test_ids = load_split()

    # Mean and standard deviation are calculated using training frames only.
    mean, std = fit_standardizer(X_train)
    X_train = standardize(X_train, mean, std)
    X_test = standardize(X_test, mean, std)

    weights, bias, history = gradient_descent(X_train, y_train)
    train_loss = half_mse_loss(predict(X_train, weights, bias), y_train)
    test_labels = predict_label(predict(X_test, weights, bias))
    accuracy = np.mean(test_labels == y_test)

    with open(PREDICTIONS_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["clip_id", "actual", "predicted"])
        for clip_id, actual, predicted in zip(test_ids, y_test, test_labels):
            writer.writerow([clip_id, LABEL_TO_SPEAKER[int(actual)],
                             LABEL_TO_SPEAKER[int(predicted)]])

    print("Train frames:", len(y_train), "| Test frames:", len(y_test))
    print("Iterations:", len(history))
    print(f"Loss: {history[0]:.6f} -> {train_loss:.6f}")
    print("Weights:", weights, "| Bias:", round(bias, 5))
    print(f"Test accuracy (same recordings): {accuracy:.2%}")
    print("Predictions saved to", PREDICTIONS_CSV)

    # Numerical Methods: compare how learning rate affects the loss.
    print("\nLearning rate comparison (100 iterations):")
    for rate in (0.01, 0.1, 1.6):
        w, b, _ = gradient_descent(X_train, y_train,
                                   learning_rate=rate, max_epochs=100,
                                   tolerance=0)
        loss = half_mse_loss(predict(X_train, w, b), y_train)
        print(f"  lr = {rate}: final loss = {loss:.6g}")


if __name__ == "__main__":
    main()
