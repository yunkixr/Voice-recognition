import argparse
import csv
import sys
from pathlib import Path


LABELS = ("A", "B")
LABEL_MAP = {"A": "A", "0": "A", "B": "B", "1": "B"}


def load_predictions(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {"clip_id", "actual", "predicted"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Invalid CSV columns in {path}. "
                f"Required columns: {', '.join(sorted(required))}"
            )

        records = []

        for row_number, row in enumerate(reader, start=2):
            clip_id = (row.get("clip_id") or "").strip()

            if not clip_id:
                raise ValueError(f"Missing clip_id at row {row_number} in {path}")

            try:
                actual_value = (row.get("actual") or "").strip().upper()
                predicted_value = (row.get("predicted") or "").strip().upper()
                actual = LABEL_MAP[actual_value]
                predicted = LABEL_MAP[predicted_value]
            except KeyError:
                raise ValueError(
                    f"Invalid label at row {row_number} in {path}. "
                    "Labels must be A, B, 0, or 1."
                )

            records.append({
                "clip_id": clip_id,
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted
            })

    if not records:
        raise ValueError(f"No prediction records found in {path}")

    return records


def calculate_metrics(records):
    matrix = {actual: {predicted: 0 for predicted in LABELS}
              for actual in LABELS}

    for row in records:
        matrix[row["actual"]][row["predicted"]] += 1

    total = len(records)
    correct = sum(matrix[label][label] for label in LABELS)
    classes = {}

    for label in LABELS:
        other = "B" if label == "A" else "A"

        tp = matrix[label][label]
        fp = matrix[other][label]
        fn = matrix[label][other]

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        classes[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": tp + fn
        }

    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "matrix": matrix,
        "classes": classes
    }


def save_csv(path, headers, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def save_results(records, metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    save_csv(
        output / "results.csv",
        ["clip_id", "actual", "predicted", "correct"],
        [
            [row["clip_id"], row["actual"], row["predicted"], row["correct"]]
            for row in records
        ]
    )

    save_csv(
        output / "error_analysis.csv",
        ["clip_id", "actual", "predicted", "error_type"],
        [
            [
                row["clip_id"],
                row["actual"],
                row["predicted"],
                f"Actual {row['actual']} predicted as {row['predicted']}"
            ]
            for row in records if not row["correct"]
        ]
    )

    rows = [
        ["total_samples", metrics["total"]],
        ["correct_predictions", metrics["correct"]],
        ["incorrect_predictions", metrics["incorrect"]],
        ["accuracy", metrics["accuracy"]]
    ]

    for label, data in metrics["classes"].items():
        for key, value in data.items():
            rows.append([f"{label}_{key}", value])

    save_csv(output / "metrics_summary.csv", ["metric", "value"], rows)

    matrix = metrics["matrix"]
    save_csv(
        output / "confusion_matrix.csv",
        ["actual/predicted", "A", "B"],
        [
            ["A", matrix["A"]["A"], matrix["A"]["B"]],
            ["B", matrix["B"]["A"], matrix["B"]["B"]]
        ]
    )


def print_report(metrics):
    print("\nEVALUATION RESULTS")
    print("------------------")
    print(f"Total samples : {metrics['total']}")
    print(f"Correct       : {metrics['correct']}")
    print(f"Incorrect     : {metrics['incorrect']}")
    print(f"Accuracy      : {metrics['accuracy']:.2%}")

    print("\nCONFUSION MATRIX")
    print("              Predicted A   Predicted B")

    for label in LABELS:
        print(
            f"Actual {label}      "
            f"{metrics['matrix'][label]['A']:>10}   "
            f"{metrics['matrix'][label]['B']:>10}"
        )

    print("\nPER-CLASS METRICS")

    for label, data in metrics["classes"].items():
        print(
            f"{label}: Precision={data['precision']:.2%}, "
            f"Recall={data['recall']:.2%}, "
            f"F1={data['f1_score']:.2%}, "
            f"Support={data['support']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate A/B voice recognition predictions."
    )
    parser.add_argument(
        "--predictions",
        default="output/predictions.csv",
        help="CSV generated by speaker_recognition.py train"
    )
    parser.add_argument(
        "--output-dir",
        default="output/evaluation"
    )
    args = parser.parse_args()

    try:
        records = load_predictions(args.predictions)
        metrics = calculate_metrics(records)

        save_results(records, metrics, args.output_dir)
        print_report(metrics)

        print(f"\nFiles saved in: {Path(args.output_dir).resolve()}")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
