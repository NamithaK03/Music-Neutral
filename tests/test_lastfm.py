import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from scipy.sparse import csr_matrix
except ImportError:
    csr_matrix = None

from sonar.lastfm import _rank, evaluate_lastfm, evaluate_lastfm_runs


class LastFmTests(unittest.TestCase):
    def test_ranking_excludes_history_and_breaks_ties_by_artist_index(self):
        ranked = _rank(
            scores=np.array([0.0, 1.0, 1.0, 0.0]),
            history=np.array([1]),
            cutoff=10,
        )
        self.assertEqual(ranked.tolist(), [2, 0, 3])

    @unittest.skipIf(csr_matrix is None, "install the optional recommendation-study dependency")
    def test_runs_deterministic_user_disjoint_interaction_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            listening_path = root / "user_artists.dat"
            with listening_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerow(("userID", "artistID", "weight"))
                for user_id in range(4):
                    for offset in range(6):
                        writer.writerow((user_id, offset + user_id * 2, 20 - offset))

            tags_path = root / "user_taggedartists.dat"
            with tags_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerow(("userID", "artistID", "tagID", "day", "month", "year"))
                for user_id in range(4):
                    for offset in range(6):
                        artist_id = offset + user_id * 2
                        writer.writerow((user_id, artist_id, offset % 3, 1, 1, 2011))

            first = evaluate_lastfm(listening_path, tags_path, seed=17)
            second = evaluate_lastfm(listening_path, tags_path, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first["counts"]["users_evaluated"], 4)
        self.assertEqual(first["counts"]["candidate_artists"], 11)
        self.assertEqual(first["counts"]["test_interactions"], 4)
        self.assertEqual(
            set(first["metrics_macro_averaged_by_user"]),
            {"popularity", "tag_content", "item_item"},
        )
        for metrics in first["metrics_macro_averaged_by_user"].values():
            self.assertGreaterEqual(metrics["recall_at_10"], 0.0)
            self.assertLessEqual(metrics["recall_at_10"], 1.0)
            self.assertGreaterEqual(metrics["ndcg_at_10"], 0.0)
            self.assertLessEqual(metrics["ndcg_at_10"], 1.0)
        self.assertIn("no mood labels", first["warning"])

    @unittest.skipIf(csr_matrix is None, "install the optional recommendation-study dependency")
    def test_rejects_insufficient_users(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            listening_path = root / "user_artists.dat"
            listening_path.write_text(
                "userID\tartistID\tweight\n1\t10\t2\n",
                encoding="utf-8",
            )
            tags_path = root / "user_taggedartists.dat"
            tags_path.write_text(
                "userID\tartistID\ttagID\n1\t10\t3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no users meet"):
                evaluate_lastfm(listening_path, tags_path)

    @unittest.skipIf(csr_matrix is None, "install the optional recommendation-study dependency")
    def test_summarizes_split_variation_across_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            listening_path = root / "user_artists.dat"
            with listening_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerow(("userID", "artistID", "weight"))
                for user_id in range(4):
                    for offset in range(6):
                        writer.writerow((user_id, offset + user_id * 2, 20 - offset))
            tags_path = root / "user_taggedartists.dat"
            with tags_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerow(("userID", "artistID", "tagID"))
                for user_id in range(4):
                    for offset in range(6):
                        writer.writerow((user_id, offset + user_id * 2, offset % 3))

            result = evaluate_lastfm_runs(
                listening_path, tags_path, seed=5, seeds=2
            )

        self.assertEqual(result["seed_range"], [5, 6])
        self.assertEqual(len(result["per_seed_metrics"]), 2)
        self.assertIn(
            "std",
            result["aggregate_metrics_macro_averaged_by_user"]["item_item"][
                "ndcg_at_10"
            ],
        )


if __name__ == "__main__":
    unittest.main()
