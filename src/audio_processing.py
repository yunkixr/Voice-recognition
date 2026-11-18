
import wave
import struct


# 1. READ AUDIO FILE
def read_audio(file_path):

    with wave.open(file_path, "rb") as audio:

        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        total_frames = audio.getnframes()

        if sample_width != 2:
            raise ValueError("Audio must be 16-bit PCM WAV.")

        if audio.getcomptype() != "NONE":
            raise ValueError("Audio must be uncompressed PCM.")

        raw_data = audio.readframes(total_frames)

    total_samples = len(raw_data) // 2

    samples = struct.unpack(
        "<" + "h" * total_samples,
        raw_data
    )

    # Convert stereo into mono
    if channels == 2:

        mono_samples = []

        for i in range(0, len(samples), 2):

            average = (samples[i] + samples[i + 1]) / 2
            mono_samples.append(average)

        samples = mono_samples

    elif channels != 1:
        raise ValueError("Only mono and stereo audio are supported.")

    return list(samples), sample_rate


# 2. NORMALIZE AUDIO
def normalize_audio(samples):

    if not samples:
        return []

    maximum = max(abs(sample) for sample in samples)

    if maximum == 0:
        return [0.0] * len(samples)

    normalized = []

    for sample in samples:
        normalized.append(sample / maximum)

    return normalized


# 3. DIVIDE AUDIO INTO FRAMES
def divide_into_frames(
    samples,
    sample_rate,
    frame_duration=0.03,
    overlap=0.5
):

    frame_size = int(sample_rate * frame_duration)
    hop_size = int(frame_size * (1 - overlap))

    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("Invalid frame configuration.")

    frames = []

    for start in range(
        0,
        len(samples) - frame_size + 1,
        hop_size
    ):

        frame = samples[start:start + frame_size]
        frames.append(frame)

    return frames
