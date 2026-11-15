import tempfile
import unittest
import zipfile
from pathlib import Path

from sonar.deam import evaluate_deam


_ANNOTATION_ROOT = "annotations/annotations averaged per song/song_level/"
_ANNOTATION_MEMBERS = (
    _ANNOTATION_ROOT + "static_annotations_averaged_songs_1_2000.csv",
    _ANNOTATION_ROOT + "static_annotations_averaged_songs_2000_2058.csv",
)


def _write_archives(root: Path, track_count: int, missing_last_feature: bool = False):
    annotations_zip = root / "annotations.zip"
    features_zip = root / "features.zip"
    with zipfile.ZipFile(annotations_zip, "w") as archive:
        rows = ["song_id, valence_mean, valence_std, arousal_mean, arousal_std"]
        rows.extend(
            "{},{:.4f},0,{:.4f},0".format(track_id, 2 + 0.2 * track_id, 8 - 0.1 * track_id)
            for track_id in range(1, track_count + 1)
        )
        archive.writestr(_ANNOTATION_MEMBERS[0], "\n".join(rows) + "\n")
        archive.writestr(
            _ANNOTATION_MEMBERS[1],
            "song_id, valence_mean, valence_std, arousal_mean, arousal_std\n",
        )

    with zipfile.ZipFile(features_zip, "w") as archive:
        for track_id in range(1, track_count + 1):
            if missing_last_feature and track_id == track_count:
                continue
            contents = (
                "frameTime;feature_a;feature_b\n"
                "0.0;{};{}\n"
                "1.0;{};{}\n".format(
                    track_id, 2 * track_id, track_id + 1, 2 * track_id + 2
                )
            )
            archive.writestr("features/{}.csv".format(track_id), contents)
    return annotations_zip, features_zip


class DeamEvaluationTests(unittest.TestCase):
    def test_song_disjoint_cross_validation_beats_training_mean_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations_zip, features_zip = _write_archives(Path(directory), 20)
            result = evaluate_deam(
                annotations_zip,
                features_zip,
                folds=5,
                seed=42,
                alpha=0.1,
            )

        self.assertEqual(result["tracks"], 20)
        self.assertEqual(result["feature_columns"], 2)
        for target in ("valence", "arousal"):
            self.assertLess(
                result["metrics"]["ridge"][target]["rmse"],
                result["metrics"]["training_mean_baseline"][target]["rmse"],
            )

    def test_missing_track_features_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations_zip, features_zip = _write_archives(
                Path(directory), 5, missing_last_feature=True
            )
            with self.assertRaisesRegex(ValueError, "missing DEAM features for track 5"):
                evaluate_deam(annotations_zip, features_zip, folds=2)


if __name__ == "__main__":
    unittest.main()
