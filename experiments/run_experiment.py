import csv

import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    FEATURE_NAMES,
    SPEAKER_TO_LABEL,
    LABEL_TO_SPEAKER,
    LEARNING_RATE
)
from src.model import (
    predict,
    half_mse_loss,
    fit_standardizer,
    standardize,
    predict_label
)
from src.gradient_descent import gradient_descent


FEATURE_CSV = "output/audio_features.csv"
PREDICTIONS_CSV = "output/predictions.csv"

# Aurora's CSV uses "pitch" for the feature that config calls "pitch_hz".
CSV_COLUMN = {"pitch_hz": "pitch", "zcr": "zcr", "rms": "rms"}
RECORDING_TO_SPEAKER = {"Voice_A": "A", "Voice_B": "B"}

SILENCE_RMS = 0.01
TRAIN_PART = 0.7

# Frames overlap by 50%, so neighbouring frames share audio.
# Skipping 20 frames (0.3 s) after the cut keeps train and test audio apart.
GAP_FRAMES = 20

# 1.6 is just above the largest stable step for this data (about 1.53), so it diverges.
LEARNING_RATES = [0.01, 0.1, 1.6]
COMPARISON_EPOCHS = 100


def load_split():
    with open(FEATURE_CSV, newline="") as file:
        rows = list(csv.DictReader(file))

    columns = [CSV_COLUMN[name] for name in FEATURE_NAMES]
    X_train, y_train = [], []
    X_test, y_test, test_ids = [], [], []

    for recording, speaker in RECORDING_TO_SPEAKER.items():
        label = SPEAKER_TO_LABEL[speaker]
        recording_rows = [row for row in rows if row["speaker"] == recording]

        # Time-based split: the first 70% of the recording trains, the end tests.
        cut = int(TRAIN_PART * len(recording_rows))

        for row in recording_rows:
            if float(row["rms"]) < SILENCE_RMS:
                continue

            features = [float(row[column]) for column in columns]
            frame = int(row["frame"])

            if frame <= cut:
                X_train.append(features)
                y_train.append(label)
            elif frame > cut + GAP_FRAMES:
                X_test.append(features)
                y_test.append(label)
                test_ids.append(f"{speaker}_frame{frame}")

    return (np.array(X_train), np.array(y_train, dtype=float),
            np.array(X_test), np.array(y_test, dtype=float), test_ids)


def accuracy(X, y, weights, bias):
    labels = predict_label(predict(X, weights, bias))
    return np.mean(labels == y)


def save_predictions(test_ids, y_test, test_labels):
    with open(PREDICTIONS_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["clip_id", "actual", "predicted"])
        for clip_id, actual, predicted in zip(test_ids, y_test, test_labels):
            writer.writerow([clip_id, LABEL_TO_SPEAKER[int(actual)],
                             LABEL_TO_SPEAKER[int(predicted)]])


def main():
    X_train_raw, y_train, X_test_raw, y_test, test_ids = load_split()

    print("Training frames:", len(y_train),
          f"(A = {int(np.sum(y_train == 0))}, B = {int(np.sum(y_train == 1))})")
    print("Test frames    :", len(y_test),
          f"(A = {int(np.sum(y_test == 0))}, B = {int(np.sum(y_test == 1))})")

    # Mean and std come from the training frames only.
    mean, std = fit_standardizer(X_train_raw)
    X_train = standardize(X_train_raw, mean, std)
    X_test = standardize(X_test_raw, mean, std)

    print("\nStandardization (from training data)")
    for name, m, s in zip(FEATURE_NAMES, mean, std):
        print(f"  {name:<8} mean = {m:10.4f}   std = {s:10.4f}")

    weights, bias, loss_history = gradient_descent(X_train, y_train)
    final_loss = half_mse_loss(predict(X_train, weights, bias), y_train)

    print("\nGradient Descent (learning rate and epochs from config.py)")
    print("  Iterations run:", len(loss_history))
    print(f"  Loss at start : {loss_history[0]:.6f}")
    print(f"  Loss at end   : {final_loss:.6f}")
    for name, w in zip(FEATURE_NAMES, weights):
        print(f"  w_{name:<8} = {w:9.5f}")
    print(f"  bias       = {bias:9.5f}")

    # Verification only: the exact minimum of the same loss.
    X_with_ones = np.column_stack([X_train, np.ones(len(X_train))])
    exact = np.linalg.lstsq(X_with_ones, y_train, rcond=None)[0]
    exact_loss = half_mse_loss(X_with_ones @ exact, y_train)

    print("\nCheck against np.linalg.lstsq (not used for training)")
    print("  lstsq weights:", np.round(exact[:3], 5), " bias:", round(exact[3], 5))
    print(f"  lstsq loss   : {exact_loss:.6f}")
    print(f"  largest difference in parameters: "
          f"{np.max(np.abs(np.append(weights, bias) - exact)):.2e}")

    test_labels = predict_label(predict(X_test, weights, bias))
    save_predictions(test_ids, y_test, test_labels)

    print("\nDiagnostic accuracy (same recordings, not real generalization)")
    print(f"  train: {accuracy(X_train, y_train, weights, bias):.2%}")
    print(f"  test : {accuracy(X_test, y_test, weights, bias):.2%}")
    print("  Test predictions saved to", PREDICTIONS_CSV)

    plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Half-MSE loss J")
    plt.title(f"Training loss, learning rate = {LEARNING_RATE}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/loss_curve.png", dpi=120)

    print(f"\nLearning rate comparison ({COMPARISON_EPOCHS} iterations each)")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    for lr in LEARNING_RATES:
        w, b, history = gradient_descent(
            X_train, y_train, learning_rate=lr,
            max_epochs=COMPARISON_EPOCHS, tolerance=0
        )
        print(f"  lr = {lr:<5} loss after {len(history)} iterations: {history[-1]:.6g}")

        if lr == LEARNING_RATES[-1]:
            right.plot(history, color="tab:red", label=f"lr = {lr}")
        else:
            left.plot(history, label=f"lr = {lr}")

    left.set_title("Small and medium learning rate")
    right.set_title("Learning rate too large")
    right.set_yscale("log")
    for ax in (left, right):
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Half-MSE loss J")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("figures/learning_rates.png", dpi=120)
    print("\nFigures saved in figures/")


if __name__ == "__main__":
    main()
