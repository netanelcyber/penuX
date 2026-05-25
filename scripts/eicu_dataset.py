"""Utilities for working with the eICU Collaborative Research Database.

Convert an eICU CSV export into this project's clinical feature schema.
Supports optional auto-fetch of the public eICU demo patient table.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import gzip
import shutil
import urllib.request


CLINICAL_COLUMNS = ["age", "heart_rate", "resp_rate", "temperature", "wbc", "bun"]

EICU_ALIASES = {
    "age": ["age"],
    "heart_rate": ["heartrate", "heart_rate", "hr"],
    "resp_rate": ["respiratoryrate", "resp_rate", "rr"],
    "temperature": ["temperature", "temp", "temperaturec"],
    "wbc": ["wbc", "whitebloodcell", "white_blood_cells"],
    "bun": ["bun", "bloodureanitrogen", "blood_urea_nitrogen"],
}

EICU_DEMO_PATIENT_CSV_GZ = "https://physionet.org/files/eicu-crd-demo/2.0/patient.csv.gz"


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    lower_to_real = {c.lower(): c for c in columns}
    for alias in aliases:
        col = lower_to_real.get(alias.lower())
        if col is not None:
            return col
    return None


def fetch_eicu_csv(url: str, output_csv: Path) -> Path:
    """Download eICU CSV (or CSV.GZ) to `output_csv`.

    If the URL ends with `.gz`, it is transparently decompressed.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if url.lower().endswith(".gz"):
        with urllib.request.urlopen(url) as src:
            with gzip.GzipFile(fileobj=src) as gz:
                with output_csv.open("wb") as out:
                    shutil.copyfileobj(gz, out)
    else:
        with urllib.request.urlopen(url) as src:
            with output_csv.open("wb") as out:
                shutil.copyfileobj(src, out)

    return output_csv


def convert_eicu_to_clinical(input_csv: Path, output_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    source_map = {target: _find_column(columns, aliases) for target, aliases in EICU_ALIASES.items()}

    out_rows: list[dict[str, str]] = []
    for row in rows:
        out_row = {}
        for target in CLINICAL_COLUMNS:
            src = source_map[target]
            out_row[target] = "" if src is None else row.get(src, "")
        out_rows.append(out_row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLINICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    return out_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert eICU table to clinical.csv schema")
    parser.add_argument("--input", "-i", type=Path, default=None, help="Path to eICU CSV file")
    parser.add_argument("--output", "-o", type=Path, default=Path("clinical_eicu.csv"), help="Output CSV")
    parser.add_argument("--autofetch", action="store_true", help="Auto-download eICU demo patient table if --input is not set")
    parser.add_argument("--fetch_url", type=str, default=EICU_DEMO_PATIENT_CSV_GZ, help="Source URL used with --autofetch")
    parser.add_argument("--fetched_input", type=Path, default=Path("eicu_patient_autofetch.csv"), help="Where to store downloaded source CSV")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to eICU CSV file")
    parser.add_argument("--output", "-o", type=Path, default=Path("clinical_eicu.csv"), help="Output CSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_csv = args.input
    if input_csv is None:
        if not args.autofetch:
            raise ValueError("Provide --input, or use --autofetch to download eICU demo data")
        input_csv = fetch_eicu_csv(args.fetch_url, args.fetched_input)
        print(f"[INFO] Downloaded eICU source -> {input_csv}")

    converted = convert_eicu_to_clinical(input_csv, args.output)
    converted = convert_eicu_to_clinical(args.input, args.output)
    print(f"[INFO] Converted {len(converted)} rows -> {args.output}")


if __name__ == "__main__":
    main()
