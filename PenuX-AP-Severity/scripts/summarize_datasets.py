"""Summarize a dataset: shape, columns, missingness, target distribution.

Usage:
    python scripts/summarize_datasets.py --data data/public_sanitized/file.csv --target-column severe
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.datasets import load_dataset, detect_target_column
from penux_ap.labels import describe_target
from penux_ap.preprocessing import summarize_missingness
from penux_ap.utils import setup_logging

log = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Summarize a sanitized dataset.")
    parser.add_argument("--data", required=True, help="Path to dataset file.")
    parser.add_argument("--target-column", default=None, help="Target column name.")
    args = parser.parse_args()

    df = load_dataset(args.data)
    print(f"\n=== Dataset: {Path(args.data).name} ===")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col} [{df[col].dtype}]")

    miss = summarize_missingness(df)
    if miss.empty:
        print("\nMissingness: none")
    else:
        print(f"\nMissingness ({len(miss)} columns):")
        print(miss.to_string())

    try:
        target_col = args.target_column or detect_target_column(df)
        print(f"\nDetected target column: '{target_col}'")
        summary = describe_target(df[target_col])
        print(f"  Total: {summary['n_total']}, Missing: {summary['n_missing']}, Classes: {summary['n_classes']}")
        print(f"  Value counts: {summary['value_counts']}")
        if summary["positive_rate"] is not None:
            print(f"  Positive rate: {summary['positive_rate']:.3f}")
    except ValueError as e:
        print(f"\nTarget detection: {e}")


if __name__ == "__main__":
    main()
