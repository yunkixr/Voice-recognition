"""
Evaluation
Expected input CSV columns:
    clip_id,actual,predicted


Labels written A/B or 0/1:
    A=0, B=1

Example:
    python evaluate.py predictions.csv
"""

import argparse
import csv
import sys
from pathlib import Path


LABEL_TO_NUMBER = {
    "A": 0,
    "B": 1,
    "0": 0,
    "1": 1,
}

NUMBER_TO_LABEL = {
    0: "A",
    1: "B",
}


def convert_label(value, row_number, column_name):
    """Convert A/B or 0/1 into the project's numeric labels."""
    cleaned = str(value).strip().upper()

    if cleaned not in LABEL_TO_NUMBER:
        raise ValueError(
            f"Invalid {column_name} value on CSV row {row_number}: "
            f"{value!r}. Use A, B, 0, or 1."
        )

    return LABEL_TO_NUMBER[cleaned]


def read_predictions(csv_path):
    """Read actual and predicted labels from a CSV file."""
    records = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required_columns = {"clip_id", "actual", "predicted"}
        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            clip_id = (row.get("clip_id") or "").strip()

            if not clip_id:
                raise ValueError(
                    f"Missing clip_id on CSV row {row_number}."
                )

            actual = convert_label(row.get("actual"), row_number, "actual")
            predicted = convert_label(
                row.get("predicted"), row_number, "predicted"
            )

            records.append({
                "clip_id": clip_id,
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted,
            })

    if not records:
        raise ValueError("The input CSV contains no prediction records.")

    return records


def calculate_accuracy(records):
    """Calculate accuracy as correct predictions divided by all predictions."""
    correct_count = sum(record["correct"] for record in records)
    return correct_count / len(records)
