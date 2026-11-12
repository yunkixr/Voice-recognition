import unittest

from src.config import (
    SAMPLE_RATE,
    FEATURE_NAMES,
    SPEAKER_TO_LABEL
)


class TestProjectConfig(unittest.TestCase):

    def test_sample_rate(self):
        self.assertEqual(SAMPLE_RATE, 8000)

    def test_feature_order(self):
        self.assertEqual(
            FEATURE_NAMES,
            ("pitch_hz", "zcr", "rms")
        )

    def test_speaker_labels(self):
        self.assertEqual(
            SPEAKER_TO_LABEL["A"], 0
        )
        self.assertEqual(
            SPEAKER_TO_LABEL["B"], 1
        )


if __name__ == "__main__":
    unittest.main()
