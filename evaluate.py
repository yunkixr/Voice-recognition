#!/usr/bin/env python3
"""
Evaluate two-speaker voice-recognition predictions using only Python's
standard library.

Expected input CSV columns:
    clip_id,actual,predicted

Labels may be A/B or 0/1 (A=0, B=1).

Examples:
    python evaluate.py
    python evaluate.py --predictions predictions.csv
    python evaluate.py --pitch-only pitch_only_predictions.csv \
        --combined combined_predictions.csv

This script evaluates predictions supplied in a CSV file. It does not train
the model or extract features from audio recordings.
"""

import argparse
import csv
import sys
from pathlib import Path


LABEL_TO_NAME = {
    "A": "A",
    "0": "A",
    "B": "B",
    "1": "B",
}
LABELS = ("A", "B")


def normalize_label(value, column_name, row_number):
    """Convert A/B or 0/1 into the standard A/B labels."""
    key = str(value).strip().upper()

    if key not in LABEL_TO_NAME:
        raise ValueError(
            f"Invalid {column_name!r} value {value!r} on CSV row {row_number}. "
            "Use A/B or 0/1 (A=0, B=1)."
        )

    return LABEL_TO_NAME[key]


def load_predictions(csv_path):
    """Load and validate a prediction CSV."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    records = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"{csv_path} is empty or has no CSV header.")

        required_columns = {"clip_id", "actual", "predicted"}
        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s): "
                f"{', '.join(sorted(missing))}. "
                "Required columns: clip_id, actual, predicted."
            )

        for row_number, row in enumerate(reader, start=2):
            clip_id = (row.get("clip_id") or "").strip()

            if not clip_id:
                raise ValueError(
                    f"Missing clip_id on CSV row {row_number} in {csv_path}."
                )

            actual = normalize_label(row.get("actual"), "actual", row_number)
            predicted = normalize_label(
                row.get("predicted"), "predicted", row_number
            )

            records.append(
                {
                    "clip_id": clip_id,
                    "actual": actual,
                    "predicted": predicted,
                    "correct": actual == predicted,
                }
            )

    if not records:
        raise ValueError(f"No prediction records found in {csv_path}.")

    return records


def calculate_metrics(records):
    """Calculate accuracy, per-class metrics, and a 2x2 confusion matrix."""
    matrix = {
        "A": {"A": 0, "B": 0},
        "B": {"A": 0, "B": 0},
    }

    for record in records:
        matrix[record["actual"]][record["predicted"]] += 1

    total = len(records)
    correct = matrix["A"]["A"] + matrix["B"]["B"]
    accuracy = correct / total

    class_metrics = {}

    for label in LABELS:
        other_label = "B" if label == "A" else "A"

        true_positive = matrix[label][label]
        false_positive = matrix[other_label][label]
        false_negative = matrix[label][other_label]

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": matrix[label]["A"] + matrix[label]["B"],
        }

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "class_metrics": class_metrics,
    }


def write_results(records, output_path):
    """Write each clip's actual label, prediction, and correctness."""
    fields = ["clip_id", "actual", "predicted", "correct"]

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "clip_id": record["clip_id"],
                    "actual": record["actual"],
                    "predicted": record["predicted"],
                    "correct": record["correct"],
                }
            )


def write_error_analysis(records, output_path):
    """Write only the incorrectly classified clips."""
    fields = ["clip_id", "actual", "predicted", "error_type"]

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for record in records:
            if not record["correct"]:
                writer.writerow(
                    {
                        "clip_id": record["clip_id"],
                        "actual": record["actual"],
                        "predicted": record["predicted"],
                        "error_type": (
                            f"Actual {record['actual']} predicted as "
                            f"{record['predicted']}"
                        ),
                    }
                )


def write_metrics_summary(metrics, output_path):
    """Write overall and per-class metrics."""
    rows = [
        ("total_samples", metrics["total"]),
        ("correct_predictions", metrics["correct"]),
        ("incorrect_predictions", metrics["incorrect"]),
        ("accuracy", f"{metrics['accuracy']:.6f}"),
    ]

    for label in LABELS:
        result = metrics["class_metrics"][label]
        rows.extend(
            [
                (f"{label}_precision", f"{result['precision']:.6f}"),
                (f"{label}_recall", f"{result['recall']:.6f}"),
                (f"{label}_f1_score", f"{result['f1_score']:.6f}"),
                (f"{label}_support", result["support"]),
            ]
        )

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_confusion_matrix(metrics, output_path):
    """Save the confusion matrix as a CSV file."""
    matrix = metrics["confusion_matrix"]

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", "A", "B"])
        writer.writerow(["A", matrix["A"]["A"], matrix["A"]["B"]])
        writer.writerow(["B", matrix["B"]["A"], matrix["B"]["B"]])


def write_experiment_comparison(pitch_metrics, combined_metrics, output_path):
    """Save accuracy results for two feature configurations."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["experiment", "features", "total_samples", "accuracy"]
        )
        writer.writerow(
            [
                "pitch_only",
                "pitch",
                pitch_metrics["total"],
                f"{pitch_metrics['accuracy']:.6f}",
            ]
        )
        writer.writerow(
            [
                "pitch_zcr_rms",
                "pitch + ZCR + RMS",
                combined_metrics["total"],
                f"{combined_metrics['accuracy']:.6f}",
            ]
        )


def print_report(metrics):
    """Print a readable evaluation summary."""
    matrix = metrics["confusion_matrix"]

    print("\nEVALUATION RESULTS")
    print("------------------")
    print(f"Total samples : {metrics['total']}")
    print(f"Correct       : {metrics['correct']}")
    print(f"Incorrect     : {metrics['incorrect']}")
    print(f"Accuracy      : {metrics['accuracy']:.2%}")

    print("\nConfusion Matrix (rows = actual, columns = predicted)")
    print("              Predicted A   Predicted B")
    print(
        f"Actual A      {matrix['A']['A']:>10}   "
        f"{matrix['A']['B']:>10}"
    )
    print(
        f"Actual B      {matrix['B']['A']:>10}   "
        f"{matrix['B']['B']:>10}"
    )

    print("\nPer-class metrics")
    for label in LABELS:
        result = metrics["class_metrics"][label]
        print(
            f"{label}: precision={result['precision']:.2%}, "
            f"recall={result['recall']:.2%}, "
            f"F1={result['f1_score']:.2%}, "
            f"support={result['support']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate A/B voice-recognition predictions."
    )
    parser.add_argument(
        "--predictions",
        default="predictions.csv",
        help="CSV with clip_id, actual, predicted (default: predictions.csv).",
    )
    parser.add_argument(
        "--pitch-only",
        help="Optional predictions CSV for the pitch-only experiment.",
    )
    parser.add_argument(
        "--combined",
        help="Optional predictions CSV for pitch + ZCR + RMS.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for generated files (default: current directory).",
    )

    args = parser.parse_args()

    if bool(args.pitch_only) != bool(args.combined):
        parser.error("--pitch-only and --combined must be supplied together.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        records = load_predictions(args.predictions)
        metrics = calculate_metrics(records)

        write_results(records, output_dir / "results.csv")
        write_error_analysis(records, output_dir / "error_analysis.csv")
        write_metrics_summary(metrics, output_dir / "metrics_summary.csv")
        write_confusion_matrix(metrics, output_dir / "confusion_matrix.csv")

        print_report(metrics)

        if args.pitch_only and args.combined:
            pitch_metrics = calculate_metrics(
                load_predictions(args.pitch_only)
            )
            combined_metrics = calculate_metrics(
                load_predictions(args.combined)
            )

            write_experiment_comparison(
                pitch_metrics,
                combined_metrics,
                output_dir / "experiment_comparison.csv",
            )

            print("\nEXPERIMENT COMPARISON")
            print("---------------------")
            print(f"Pitch only       : {pitch_metrics['accuracy']:.2%}")
            print(f"Pitch + ZCR + RMS: {combined_metrics['accuracy']:.2%}")

        print(f"\nFiles saved in: {output_dir.resolve()}")
        print(" - results.csv")
        print(" - metrics_summary.csv")
        print(" - error_analysis.csv")
        print(" - confusion_matrix.csv")

        if args.pitch_only and args.combined:
            print(" - experiment_comparison.csv")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
