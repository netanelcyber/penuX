"""Utilities for working with the eICU Collaborative Research Database.

Convert an eICU CSV export into this project's clinical feature schema.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv


CLINICAL_COLUMNS = ["age", "heart_rate", "resp_rate", "temperature", "wbc", "bun"]

EICU_ALIASES = {
    "age": ["age"],
    "heart_rate": ["heartrate", "heart_rate", "hr"],
    "resp_rate": ["respiratoryrate", "resp_rate", "rr"],
    "temperature": ["temperature", "temp", "temperaturec"],
    "wbc": ["wbc", "whitebloodcell", "white_blood_cells"],
    "bun": ["bun", "bloodureanitrogen", "blood_urea_nitrogen"],
}


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    lower_to_real = {c.lower(): c for c in columns}
    for alias in aliases:
        col = lower_to_real.get(alias.lower())
        if col is not None:
            return col
    return None


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
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to eICU CSV file")
    parser.add_argument("--output", "-o", type=Path, default=Path("clinical_eicu.csv"), help="Output CSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    converted = convert_eicu_to_clinical(args.input, args.output)
    print(f"[INFO] Converted {len(converted)} rows -> {args.output}")


if __name__ == "__main__":
    main()
