import unittest

from sonar.simulation import evaluate_fusion_rows


class EvaluationTests(unittest.TestCase):
    def test_probability_averaging_conflict_and_fallback_are_distinguished(self):
        rows = [
            {
                "true_label": "neutral",
                "text_probs": "[0.60, 0.10, 0.10, 0.20]",
                "speech_probs": "[0.10, 0.65, 0.10, 0.15]",
                "asr_confidence": "0.8",
                "text_is_asr": "false",
            }
        ]

        result = evaluate_fusion_rows(rows)

        self.assertEqual(result["meta"]["conflict_sample_count"], 1)
        self.assertEqual(result["naive"]["conflict_accuracy"], 0.0)
        self.assertEqual(result["confidence_aware"]["conflict_accuracy"], 0.0)
        self.assertEqual(result["confidence_aware_fallback"]["conflict_accuracy"], 1.0)
        self.assertEqual(result["confidence_aware_fallback"]["accuracy"], 1.0)

    def test_rejects_probability_arrays_that_do_not_sum_to_one(self):
        rows = [
            {
                "true_label": "happy",
                "text_probs": "[0.5, 0.1, 0.1, 0.1]",
                "speech_probs": "",
            }
        ]

        with self.assertRaises(ValueError):
            evaluate_fusion_rows(rows)


if __name__ == "__main__":
    unittest.main()
