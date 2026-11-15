import unittest

from sonar.fusion import FusionConfig, Prediction, fuse_emotions


class FusionTests(unittest.TestCase):
    def test_conflict_uses_mean_trust_and_more_trusted_label(self):
        text = Prediction("happy", 0.9, auxiliary_confidence=0.8)
        speech = Prediction("sad", 0.6)

        result = fuse_emotions(text, speech, asr_confidence=0.8)

        self.assertEqual(result.label, "happy")
        self.assertEqual(result.modality, "text-dominant")
        self.assertAlmostEqual(result.confidence, (0.87 + 0.66) / 2)
        self.assertFalse(result.fallback)

    def test_low_confidence_falls_back_and_gates_fine_label(self):
        text = Prediction(
            "sad", 0.4, auxiliary_confidence=0.4,
            fine_label="sadness", fine_confidence=0.99,
        )

        result = fuse_emotions(text, None)

        self.assertEqual(result.label, "neutral")
        self.assertIsNone(result.fine_label)
        self.assertTrue(result.fallback)

    def test_asr_text_branch_is_ignored_below_gate(self):
        text = Prediction("happy", 0.99, auxiliary_confidence=0.99)
        speech = Prediction("sad", 0.9)

        result = fuse_emotions(
            text, speech, asr_confidence=0.69, text_is_asr=True
        )

        self.assertEqual(result.label, "sad")
        self.assertEqual(result.modality, "speech-only")
        self.assertIsNone(result.text_trust)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            Prediction("happy", 1.1)


if __name__ == "__main__":
    unittest.main()
