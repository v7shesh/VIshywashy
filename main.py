"""
Command-line entry point for the Student Dropout Early Intervention System.

Usage:
    python -m src.main audit      # P1: profile the raw dataset
    python -m src.main train      # P2+P4: preprocess, train, evaluate, save artifacts
    python -m src.main explain    # P5: global feature importance report
    python -m src.main score CSV  # batch-score a CSV of students, print ranked risk list
"""
import argparse
import sys

import pandas as pd

from src import config
from src.utils import section, load_raw_csv, detect_target_column


def run_audit() -> None:
    section("P1 - DATA AUDIT")

    if not config.RAW_CSV_PATH.exists():
        print(f"ERROR: expected CSV at {config.RAW_CSV_PATH} - not found.")
        sys.exit(1)

    df = load_raw_csv(config.RAW_CSV_PATH)
    print(f"Loaded: {config.RAW_CSV_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    section("Columns and dtypes")
    with pd.option_context("display.max_rows", None):
        print(df.dtypes)

    section("First 5 rows")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head())

    section("Missing values per column")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.sort_values(ascending=False))

    section("Duplicate rows")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")

    target_col = detect_target_column(df)
    section(f"Target column detected: '{target_col}'")
    counts = df[target_col].value_counts()
    pct = (counts / len(df) * 100).round(2)
    dist = pd.DataFrame({"count": counts, "pct": pct})
    print(dist)

    imbalance_ratio = counts.max() / counts.min()
    print(f"\nMax/min class ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 1.5:
        print(
            "FLAG: meaningful class imbalance detected. Will use class_weight='balanced' "
            "and stratified splits/CV in P2/P4, and report macro + per-class metrics "
            "(not just accuracy) so minority-class performance is not hidden."
        )
    else:
        print("Classes are reasonably balanced.")

    section("Numeric feature summary")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.describe().T)

    section("P1 SUMMARY")
    print(
        f"- {df.shape[0]} students, {df.shape[1]} columns (target='{target_col}')\n"
        f"- Missing values: {'none' if missing.empty else missing.sum()}\n"
        f"- Duplicate rows: {df.duplicated().sum()}\n"
        f"- Class distribution: {dict(counts)}\n"
        f"- Imbalance ratio (max/min): {imbalance_ratio:.2f}x\n"
        "Review above, then continue to P2 (preprocessing)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Student Dropout Early Intervention System")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="P1: profile the raw dataset")
    sub.add_parser("train", help="P2+P4: preprocess, train, evaluate, save artifacts")
    sub.add_parser("explain", help="P5: global feature importance report")
    score_parser = sub.add_parser("score", help="Batch-score a CSV of students")
    score_parser.add_argument("csv_path", help="Path to CSV of student records to score")

    args = parser.parse_args()

    if args.command == "audit":
        run_audit()
    elif args.command == "train":
        from src.model import run_training
        run_training()
    elif args.command == "explain":
        from src.explain import run_explain_report
        run_explain_report()
    elif args.command == "score":
        from src.intervention import run_batch_score
        run_batch_score(args.csv_path)


if __name__ == "__main__":
    main()
