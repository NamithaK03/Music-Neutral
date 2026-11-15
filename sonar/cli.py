"""Command-line entry points for SONAR evaluation and simulation."""

import argparse
import csv
import json
from pathlib import Path

from sonar.deam import evaluate_deam
from sonar.lastfm import evaluate_lastfm_runs, write_lastfm_result
from sonar.meld import evaluate_meld, write_predictions_csv
from sonar.meld_checkpoints import (
    evaluate_meld_checkpoints,
    evaluate_meld_episode_disjoint,
)
from sonar.simulation import evaluate_fusion_rows, run_experiment, tune_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sonar")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="run the synthetic-user recommender simulation")
    simulate.add_argument("--seeds", type=int, default=5)
    simulate.add_argument("--interactions", type=int, default=300)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--output", type=Path, default=Path("results/simulation.json"))

    tune = commands.add_parser(
        "tune", help="select recommender parameters on validation seeds and evaluate on held-out seeds"
    )
    tune.add_argument("--interactions", type=int, default=300)
    tune.add_argument("--validation-seeds", type=int, default=5)
    tune.add_argument("--validation-seed", type=int, default=42)
    tune.add_argument("--test-seeds", type=int, default=5)
    tune.add_argument("--test-seed", type=int, default=142)
    tune.add_argument("--output", type=Path, default=Path("results/tuned_simulation.json"))

    evaluate = commands.add_parser(
        "evaluate-emotion", help="score held-out four-class predictions from a CSV file"
    )
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, default=Path("results/emotion_evaluation.json"))

    deam = commands.add_parser(
        "evaluate-deam", help="evaluate music valence/arousal regression on the DEAM dataset"
    )
    deam.add_argument("--annotations-zip", type=Path, required=True)
    deam.add_argument("--features-zip", type=Path, required=True)
    deam.add_argument("--folds", type=int, default=5)
    deam.add_argument("--seed", type=int, default=42)
    deam.add_argument("--alpha", type=float, default=1.0)
    deam.add_argument("--output", type=Path, default=Path("results/deam_evaluation.json"))

    lastfm = commands.add_parser(
        "evaluate-lastfm",
        help="evaluate offline artist ranking on HetRec 2011 Last.fm interactions",
    )
    lastfm.add_argument("--listening", type=Path, required=True)
    lastfm.add_argument("--tags", type=Path, required=True)
    lastfm.add_argument("--seed", type=int, default=42)
    lastfm.add_argument("--seeds", type=int, default=5)
    lastfm.add_argument(
        "--output", type=Path, default=Path("results/lastfm_evaluation.json")
    )

    meld = commands.add_parser(
        "evaluate-meld",
        help="train and evaluate linear text/audio baselines on MELD",
    )
    meld.add_argument("--train-csv", type=Path, required=True)
    meld.add_argument("--dev-csv", type=Path, required=True)
    meld.add_argument("--test-csv", type=Path, required=True)
    meld.add_argument("--audio-features", type=Path, required=True)
    meld.add_argument("--output", type=Path, default=Path("results/meld_evaluation.json"))
    meld.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/meld_predictions.csv"),
    )

    meld_models = commands.add_parser(
        "evaluate-meld-checkpoints",
        help="evaluate the released MELD text and speech baseline checkpoints",
    )
    meld_models.add_argument("--train-csv", type=Path, required=True)
    meld_models.add_argument("--dev-csv", type=Path, required=True)
    meld_models.add_argument("--test-csv", type=Path, required=True)
    meld_models.add_argument("--label-pickle", type=Path, required=True)
    meld_models.add_argument("--text-features", type=Path, required=True)
    meld_models.add_argument("--speech-features", type=Path, required=True)
    meld_models.add_argument("--text-weights", type=Path, required=True)
    meld_models.add_argument("--speech-weights", type=Path, required=True)
    meld_models.add_argument(
        "--output",
        type=Path,
        default=Path("results/meld_checkpoint_evaluation.json"),
    )
    meld_models.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/meld_checkpoint_predictions.csv"),
    )

    meld_grouped = commands.add_parser(
        "evaluate-meld-episode-disjoint",
        help="score MELD test predictions on episodes absent from train and dev",
    )
    meld_grouped.add_argument("--train-csv", type=Path, required=True)
    meld_grouped.add_argument("--dev-csv", type=Path, required=True)
    meld_grouped.add_argument("--test-csv", type=Path, required=True)
    meld_grouped.add_argument("--predictions", type=Path, required=True)
    meld_grouped.add_argument(
        "--output",
        type=Path,
        default=Path("results/meld_episode_disjoint_evaluation.json"),
    )

    args = parser.parse_args(argv)
    if args.command == "simulate":
        if args.seeds <= 0:
            parser.error("--seeds must be positive")
        result = run_experiment(
            seeds=range(args.seed, args.seed + args.seeds),
            interactions=args.interactions,
        )
    elif args.command == "tune":
        if args.validation_seeds <= 0 or args.test_seeds <= 0:
            parser.error("--validation-seeds and --test-seeds must be positive")
        validation_seeds = range(
            args.validation_seed, args.validation_seed + args.validation_seeds
        )
        test_seeds = range(args.test_seed, args.test_seed + args.test_seeds)
        if set(validation_seeds) & set(test_seeds):
            parser.error("validation and test seed ranges must not overlap")
        result = tune_experiment(
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
            interactions=args.interactions,
        )
    elif args.command == "evaluate-emotion":
        with args.input.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        result = {
            "evidence_type": "measured_on_user_supplied_predictions",
            "warning": (
                "Metrics are only as valid as the supplied held-out predictions, "
                "labels, split, and provenance."
            ),
            "input": str(args.input),
            "metrics": evaluate_fusion_rows(rows),
        }
    elif args.command == "evaluate-meld":
        result, predictions = evaluate_meld(
            args.train_csv,
            args.dev_csv,
            args.test_csv,
            args.audio_features,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_predictions_csv(args.predictions, predictions)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Wrote " + str(args.output))
        print("Wrote " + str(args.predictions))
        return
    elif args.command == "evaluate-meld-checkpoints":
        result, predictions = evaluate_meld_checkpoints(
            args.train_csv,
            args.dev_csv,
            args.test_csv,
            args.label_pickle,
            args.text_features,
            args.speech_features,
            args.text_weights,
            args.speech_weights,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_predictions_csv(args.predictions, predictions)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Wrote " + str(args.output))
        print("Wrote " + str(args.predictions))
        return
    elif args.command == "evaluate-lastfm":
        result = evaluate_lastfm_runs(
            args.listening, args.tags, seed=args.seed, seeds=args.seeds
        )
        write_lastfm_result(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Wrote " + str(args.output))
        return
    elif args.command == "evaluate-meld-episode-disjoint":
        result = evaluate_meld_episode_disjoint(
            args.train_csv, args.dev_csv, args.test_csv, args.predictions
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print("Wrote " + str(args.output))
        return
    else:
        result = evaluate_deam(
            args.annotations_zip,
            args.features_zip,
            folds=args.folds,
            seed=args.seed,
            alpha=args.alpha,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Wrote " + str(args.output))


if __name__ == "__main__":
    main()
