from src.config import FEATURE_NAMES, LEARNING_RATE
from src.model import predict, half_mse_loss, fit_standardizer, standardize, predict_label
from src.gradient_descent import gradient_descent


# Kolom CSV Aurora: speaker, frame, rms, zcr, pitch
FEATURE_COLUMNS = {"pitch_hz": "pitch", "zcr": "zcr", "rms": "rms"}


def load_data():
    with open("output/audio_features.csv", encoding="utf-8-sig") as file:
        headers = file.readline().strip().split(",")
        required = ("speaker", "frame", "rms", "zcr", "pitch")
        for name in required:
            if name not in headers:
                raise ValueError("Missing CSV column: " + name)

        rows = []
        for line_number, line in enumerate(file, start=2):
            if line.strip():
                values = line.strip().split(",")
                if len(values) != len(headers):
                    raise ValueError(f"Invalid CSV format on line {line_number}")
                rows.append(dict(zip(headers, values)))

    X_train, y_train = [], []
    X_test, y_test, test_ids = [], [], []

    for speaker, label in (("Voice_A", 0), ("Voice_B", 1)):
        speaker_rows = [row for row in rows if row["speaker"] == speaker]
        cut = int(0.7 * len(speaker_rows))

        for row in speaker_rows:
            if float(row["rms"]) < 0.01:
                continue

            X = [float(row[FEATURE_COLUMNS[name]]) for name in FEATURE_NAMES]
            frame = int(row["frame"])
            if frame <= cut:
                X_train.append(X)
                y_train.append(label)
            elif frame > cut + 20:
                X_test.append(X)
                y_test.append(label)
                test_ids.append(f"{speaker}_frame{frame}")

    return X_train, y_train, X_test, y_test, test_ids


def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual)


def main():
    X_train, y_train, X_test, y_test, test_ids = load_data()
    if not X_train or not X_test:
        raise ValueError("Training or testing data is empty")

    # Mean dan standar deviasi hanya dihitung dari training data.
    mean, std = fit_standardizer(X_train)
    X_train = standardize(X_train, mean, std)
    X_test = standardize(X_test, mean, std)

    weights, bias, history = gradient_descent(X_train, y_train)
    final_loss = half_mse_loss(predict(X_train, weights, bias), y_train)

    print("Training frames:", len(y_train))
    print("Testing frames :", len(y_test))
    print("Learning rate  :", LEARNING_RATE)
    print("Iterations     :", len(history))
    print("Initial loss   :", round(history[0], 6))
    print("Final loss     :", round(final_loss, 6))
    print("Weights        :", [round(w, 5) for w in weights])
    print("Bias           :", round(bias, 5))

    train_pred = predict_label(predict(X_train, weights, bias))
    test_pred = predict_label(predict(X_test, weights, bias))
    print("Train accuracy :", f"{accuracy(y_train, train_pred):.2%}")
    print("Test accuracy  :", f"{accuracy(y_test, test_pred):.2%}")
    print("(Test frames are from the same recordings, not new speakers.)")

    # Output untuk evaluate.py Putra.
    with open("output/predictions.csv", "w") as file:
        file.write("clip_id,actual,predicted\n")
        for i in range(len(test_ids)):
            actual = "A" if y_test[i] == 0 else "B"
            predicted = "A" if test_pred[i] == 0 else "B"
            file.write(f"{test_ids[i]},{actual},{predicted}\n")

    print("Predictions saved to output/predictions.csv")

    print("\nLearning rate comparison (100 iterations)")
    for lr in (0.01, 0.1, 1.6):
        w, b, losses = gradient_descent(X_train, y_train,
                                       learning_rate=lr, max_epochs=100,
                                       tolerance=0)
        ending = half_mse_loss(predict(X_train, w, b), y_train)
        print(f"lr = {lr}: loss = {ending:.6g}")


if __name__ == "__main__":
    main()
