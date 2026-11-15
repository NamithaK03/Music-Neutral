import unittest

from sonar.metrics import accuracy, expected_calibration_error, macro_f1


class MetricTests(unittest.TestCase):
    def test_known_four_class_metrics(self):
        truth = ["happy", "sad", "angry", "neutral"]
        predicted = ["happy", "sad", "angry", "happy"]

        self.assertEqual(accuracy(truth, predicted), 0.75)
        self.assertAlmostEqual(macro_f1(truth, predicted), 2 / 3)
        self.assertAlmostEqual(
            expected_calibration_error(
                truth, predicted, [0.9, 0.9, 0.9, 0.9], bins=10
            ),
            0.15,
        )

    def test_rejects_mismatched_metric_inputs(self):
        with self.assertRaises(ValueError):
            accuracy(["happy"], [])


if __name__ == "__main__":
    unittest.main()
