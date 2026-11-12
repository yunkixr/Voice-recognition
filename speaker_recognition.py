import wave, struct, math, json, random, sys

FRAME_SEC = 0.04
LEARNING_RATE = 0.15
EPOCHS = 500
MODEL = "speaker_model.json"


def read_wav(path):
    with wave.open(path, "rb") as audio:
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        rate = audio.getframerate()
        raw = audio.readframes(audio.getnframes())

    if width != 2:
        raise ValueError(path + ": use 16-bit PCM WAV")

    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    samples = [
        sum(values[i:i + channels]) / (channels * 32768)
        for i in range(0, len(values), channels)
    ]
    return samples, rate


def features(frame, rate):
    """Five hand-made voice features: loudness, shape, crossings and pitch."""
    n = len(frame)
    rms = math.sqrt(sum(x * x for x in frame) / n)
    mean_abs = sum(abs(x) for x in frame) / n
    peak = max(abs(x) for x in frame)
    zcr = sum((frame[i - 1] >= 0) != (frame[i] >= 0)
              for i in range(1, n)) / (n - 1)

    # Estimate pitch using autocorrelation (no audio-processing library).
    best_lag, best_score = 0, 0
    for lag in range(max(1, rate // 400), min(rate // 70, n // 2)):
        score = sum(frame[i] * frame[i + lag] for i in range(n - lag))
        if score > best_score:
            best_lag, best_score = lag, score
    pitch = rate / best_lag if best_lag and best_score > 0 else 0

    return [rms, mean_abs, peak, zcr, pitch]


def extract(samples, rate):
    size = max(256, int(FRAME_SEC * rate))
    size = 2 ** int(math.log2(size))
    frames = []
    for start in range(0, len(samples) - size + 1, size):
        frame = samples[start:start + size]
        rms = math.sqrt(sum(x * x for x in frame) / size)
        if rms > 0.01:
            frames.append((start / rate, features(frame, rate)))
    if not frames:
        raise ValueError("No speech found in audio")
    return frames


def sigmoid(z):
    z = max(-60, min(60, z))
    return 1 / (1 + math.exp(-z))


def scale_fit(data):
    count = len(data)
    mean = [sum(row[j] for row in data) / count for j in range(5)]
    std = [math.sqrt(sum((row[j] - mean[j]) ** 2 for row in data) / count)
           or 1 for j in range(5)]
    return mean, std


def scale(row, mean, std):
    return [(row[j] - mean[j]) / std[j] for j in range(5)]


def probability(weights, bias, x):
    return sigmoid(bias + sum(w * value for w, value in zip(weights, x)))


def train(X, y):
    weights, bias = [0.0] * 5, 0.0
    for epoch in range(EPOCHS):
        gradients, grad_bias = [0.0] * 5, 0.0
        for x, target in zip(X, y):
            error = probability(weights, bias, x) - target
            for j in range(5):
                gradients[j] += error * x[j]
            grad_bias += error

        count = len(X)
        for j in range(5):
            weights[j] -= LEARNING_RATE * gradients[j] / count
        bias -= LEARNING_RATE * grad_bias / count

        if epoch % 100 == 0:
            print("Epoch", epoch, "completed")
    return weights, bias


def train_command(file_a, file_b):
    names, data, labels = [file_a, file_b], [], []
    rate_used = None

    for label, path in enumerate(names):
        samples, rate = read_wav(path)
        if rate_used is not None and rate != rate_used:
            raise ValueError("Both training WAV files must use the same sample rate")
        rate_used = rate
        rows = [row for _, row in extract(samples, rate)]
        data.extend(rows)
        labels.extend([label] * len(rows))
        print(path, ":", len(rows), "speech frames")

    # Balance the two speakers so one recording does not dominate training.
    random.seed(1)
    groups = [[x for x, y in zip(data, labels) if y == label] for label in (0, 1)]
    count = min(map(len, groups))
    training = [(x, label) for label, group in enumerate(groups)
                for x in random.sample(group, count)]
    random.shuffle(training)

    X_raw = [x for x, _ in training]
    y = [label for _, label in training]
    mean, std = scale_fit(X_raw)
    X = [scale(x, mean, std) for x in X_raw]
    weights, bias = train(X, y)

    with open(MODEL, "w") as file:
        json.dump({"names": names, "rate": rate_used, "mean": mean,
                   "std": std, "weights": weights, "bias": bias}, file)

    print("\nTraining complete. Model saved as", MODEL)
    print("Formula: probability = sigmoid(bias + w1*x1 + ... + w5*x5)")
    print("Speaker A = 0; Speaker B = 1")


def predict_command(path):
    with open(MODEL) as file:
        model = json.load(file)

    samples, rate = read_wav(path)
    if rate != model["rate"]:
        raise ValueError("Use the same sample rate as the training recordings")

    segments = {}
    for time, row in extract(samples, rate):
        x = scale(row, model["mean"], model["std"])
        segments.setdefault(int(time), []).append(
            probability(model["weights"], model["bias"], x))

    print("\nSpeaker prediction for", path)
    for second in sorted(segments):
        score = sum(segments[second]) / len(segments[second])
        speaker = model["names"][1] if score >= 0.5 else model["names"][0]
        confidence = max(score, 1 - score) * 100
        print(f"{second:>4}-{second + 1:<4} sec  {speaker}  ({confidence:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "train":
        train_command(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "predict":
        predict_command(sys.argv[2])
    else:
        print(__doc__)
