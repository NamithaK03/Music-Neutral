import unittest

import numpy as np

from sonar.recommender import GENRES, LinUCB, UserHistory, make_context
from sonar.simulation import generate_catalog, run_experiment, tune_experiment


class RecommenderTests(unittest.TestCase):
    def test_context_is_32_finite_features(self):
        context = make_context("happy", 0.8, engagement=[-1.0, 0.5])
        self.assertEqual(context.shape, (32,))
        self.assertTrue(np.all(np.isfinite(context)))

    def test_linucb_learns_reward_for_selected_action(self):
        policy = LinUCB(actions=GENRES[:2], dimension=32)
        context = make_context("happy", 0.9)
        for _ in range(5):
            policy.update("pop", context, 1.0)

        policy.exploration = 0.0
        self.assertEqual(policy.select(context, ("pop", "dance")), "pop")

    def test_simulation_is_seeded_and_emits_correct_scope(self):
        first = run_experiment(seeds=(42,), interactions=10)
        second = run_experiment(seeds=(42,), interactions=10)

        self.assertEqual(first, second)
        self.assertEqual(first["evidence_type"], "new_synthetic_simulation")
        self.assertEqual(len(generate_catalog(1)), 160)

    def test_tuner_uses_disjoint_validation_and_test_seeds(self):
        result = tune_experiment(
            validation_seeds=(7,),
            test_seeds=(8,),
            interactions=5,
        )

        self.assertEqual(result["grid_size"], 27)
        self.assertEqual(
            set(result["validation_seeds"]) & set(result["test_seeds"]), set()
        )
        self.assertIn("hybrid_linucb_tuned", result["held_out_test"])
        self.assertIn(
            result["selected_parameters"]["exploration"], (0.1, 0.5, 1.0)
        )

    def test_tuner_rejects_seed_overlap(self):
        with self.assertRaises(ValueError):
            tune_experiment(validation_seeds=(1, 2), test_seeds=(2, 3), interactions=5)


if __name__ == "__main__":
    unittest.main()
