
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
            raise ValueError(f"Invalid CSV columns in {path}")

        records = []

        for row in reader:
            clip_id = (row.get("clip_id") or "").strip()

            if not clip_id:
                raise ValueError("Missing clip_id")

            try:
                actual = LABEL_MAP[row["actual"].strip().upper()]
                predicted = LABEL_MAP[row["predicted"].strip().upper()]
            except (KeyError, AttributeError):
                raise ValueError(f"Invalid label in {path}")

            records.append({
                "clip_id": clip_id,
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted
            })

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def calculate_metrics(records):
    matrix = {a: {b: 0 for b in LABELS} for a in LABELS}

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
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

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
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def save_results(records, metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    save_csv(
        output / "results.csv",
        ["clip_id", "actual", "predicted", "correct"],
        [[r["clip_id"], r["actual"], r["predicted"], r["correct"]] for r in records]
    )

    save_csv(
        output / "error_analysis.csv",
        ["clip_id", "actual", "predicted", "error_type"],
        [[r["clip_id"], r["actual"], r["predicted"],
          f"Actual {r['actual']} predicted as {r['predicted']}"]
         for r in records if not r["correct"]]
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
        [["A", matrix["A"]["A"], matrix["A"]["B"]],
         ["B", matrix["B"]["A"], matrix["B"]["B"]]]
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
        description="Evaluate A/B voice recognition."
    )

    parser.add_argument("--predictions", default="predictions.csv")
    parser.add_argument("--pitch-only")
    parser.add_argument("--combined")
    parser.add_argument("--output-dir", default=".")

    args = parser.parse_args()

    if bool(args.pitch_only) != bool(args.combined):
        parser.error("--pitch-only and --combined must be used together")

    try:
        records = load_predictions(args.predictions)
        metrics = calculate_metrics(records)

        save_results(records, metrics, args.output_dir)
        print_report(metrics)

        if args.pitch_only and args.combined:
            pitch = calculate_metrics(load_predictions(args.pitch_only))
            combined = calculate_metrics(load_predictions(args.combined))

            save_csv(
                Path(args.output_dir) / "experiment_comparison.csv",
                ["experiment", "features", "total_samples", "accuracy"],
                [
                    ["pitch_only", "pitch", pitch["total"], pitch["accuracy"]],
                    ["pitch_zcr_rms", "pitch + ZCR + RMS",
                     combined["total"], combined["accuracy"]]
                ]
            )

            print("\nEXPERIMENT COMPARISON")
            print(f"Pitch only       : {pitch['accuracy']:.2%}")
            print(f"Pitch + ZCR + RMS: {combined['accuracy']:.2%}")

        print(f"\nFiles saved in: {Path(args.output_dir).resolve()}")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
