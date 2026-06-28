"""Sanitize datasets by removing likely identifier columns.

Usage:
    python scripts/sanitize_datasets.py --input data/raw --output data/public_sanitized
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from penux_ap.datasets import load_dataset, sanitize_identifiers
from penux_ap.config import SUPPORTED_EXTENSIONS
from penux_ap.utils import setup_logging

log = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Sanitize datasets by removing identifier columns.")
    parser.add_argument("--input", required=True, help="Input directory with raw data files.")
    parser.add_argument("--output", required=True, help="Output directory for sanitized files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--keep-column", nargs="*", default=[], help="Columns to keep even if they match identifier patterns.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        log.warning("No supported files found in %s", input_dir)
        sys.exit(0)

    report = []
    for fpath in files:
        out_path = output_dir / fpath.name
        if out_path.exists() and not args.force:
            log.warning("Output file exists (use --force to overwrite): %s", out_path)
            continue
        try:
            df = load_dataset(fpath)
            sanitized, removed = sanitize_identifiers(df, keep_columns=args.keep_column)
            sanitized.to_csv(out_path, index=False) if fpath.suffix == ".csv" else sanitized.to_excel(out_path, index=False)
            log.info("Saved sanitized file: %s (removed columns: %s)", out_path, removed)
            report.append({"file": str(fpath.name), "removed_columns": removed, "shape": list(sanitized.shape)})
        except Exception as e:
            log.error("Failed to process %s: %s", fpath, e)
            report.append({"file": str(fpath.name), "error": str(e)})

    report_path = output_dir / "sanitization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Sanitization report saved to %s", report_path)


if __name__ == "__main__":
    main()
