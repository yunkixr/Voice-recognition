"""
୨୧ Shared Configuration
Numerical Methods — Voice Recognition

Team: Regina, Aurora, Putra
"""

# Audio specifications
SAMPLE_RATE = 8000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM

# Feature extraction order
FEATURE_NAMES = (
    "pitch_hz",
    "zcr",
    "rms"
)

# Speaker labels
SPEAKER_TO_LABEL = {
    "A": 0,
    "B": 1
}

LABEL_TO_SPEAKER = {
    0: "A",
    1: "B"
}

# Numerical model defaults
LEARNING_RATE = 0.05
MAX_EPOCHS = 1500
TOLERANCE = 1e-10
