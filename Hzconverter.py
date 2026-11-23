
import wave
import numpy as np

input_file = "src/conversation/conversation.wav"
output_file = "src/conversation/conversation_converted.wav"

with wave.open(input_file, "rb") as audio:
    channels = audio.getnchannels()
    sample_width = audio.getsampwidth()
    sample_rate = audio.getframerate()
    frames = audio.readframes(audio.getnframes())

if sample_width != 2:
    raise ValueError("Audio must be 16-bit PCM WAV.")

if sample_rate != 44100:
    raise ValueError(f"Expected 44100 Hz, got {sample_rate} Hz.")

samples = np.frombuffer(frames, dtype="<i2")

if channels > 1:
    samples = samples.reshape(-1, channels)

samples = samples.astype(np.float64)

old_length = len(samples)
new_length = round(old_length * 48000 / 44100)

old_positions = np.arange(old_length)
new_positions = np.arange(new_length) * 44100 / 48000

if channels == 1:
    converted = np.interp(
        new_positions,
        old_positions,
        samples
    )
else:
    converted = np.column_stack([
        np.interp(
            new_positions,
            old_positions,
            samples[:, channel]
        )
        for channel in range(channels)
    ])

converted = np.clip(
    np.round(converted),
    -32768,
    32767
).astype("<i2")

with wave.open(output_file, "wb") as audio:
    audio.setnchannels(channels)
    audio.setsampwidth(2)
    audio.setframerate(48000)
    audio.writeframes(converted.tobytes())

print("Conversion complete!")
print("Output:", output_file)
print("Original sample rate: 44100 Hz")
print("New sample rate: 48000 Hz")
