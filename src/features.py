
import math


# 1. ROOT MEAN SQUARE
def calculate_rms(frame):

    if len(frame) == 0:
        return 0.0

    total = 0

    for sample in frame:
        total += sample ** 2

    mean_square = total / len(frame)

    rms = math.sqrt(mean_square)

    return rms


# 2. ZERO CROSSING RATE
def calculate_zcr(frame):

    if len(frame) < 2:
        return 0.0

    crossings = 0

    for i in range(len(frame) - 1):

        if frame[i] * frame[i + 1] < 0:
            crossings += 1

    zcr = crossings / (2 * len(frame))

    return zcr


# 3. PITCH USING AUTOCORRELATION

def calculate_pitch(frame, sample_rate):

    if len(frame) < 2:
        return 0.0

    # Remove the mean value
    mean = sum(frame) / len(frame)
    centered = [x - mean for x in frame]

    energy = sum(x * x for x in centered)

    if energy < 1e-8:
        return 0.0

    # Frequency range
    min_frequency = 70
    max_frequency = 400

    min_lag = int(sample_rate / max_frequency)

    max_lag = min(
        int(sample_rate / min_frequency),
        len(centered) - 1
    )

    best_correlation = float("-inf")
    best_lag = 0

    # Use every 4th sample to reduce computation
    step = 4

    for lag in range(min_lag, max_lag + 1):

        correlation = 0

        for i in range(0, len(centered) - lag, step):

            correlation += centered[i] * centered[i + lag]

        if correlation > best_correlation:

            best_correlation = correlation
            best_lag = lag

    if best_lag == 0 or best_correlation <= 0:
        return 0.0

    pitch = sample_rate / best_lag

    return pitch


def extract_features(frames, sample_rate):

    features = []

    for index, frame in enumerate(frames):

        rms = calculate_rms(frame)
        zcr = calculate_zcr(frame)
        pitch = calculate_pitch(frame, sample_rate)

        features.append({
            "frame": index + 1,
            "rms": rms,
            "zcr": zcr,
            "pitch": pitch
        })

    return features
