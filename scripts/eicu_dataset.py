"""Utilities for working with the eICU Collaborative Research Database.

Convert an eICU CSV export into this project's clinical feature schema.
Supports optional auto-fetch of the full eICU patient table (non-demo).
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

EICU_FULL_PATIENT_CSV_GZ = "https://www.kaggle.com/api/v1/datasets/download/bilal1907/mimic-iii-10k"


AMSTERDAMUMCDB_URL = "https://github.com/AmsterdamUMC/AmsterdamUMCdb"

AMSTERDAM_TO_MIMIC_ALIASES = {
    "age": ["agegroup", "age"],
    "heart_rate": ["hartfrequentie", "heart_rate", "heartrate", "hr"],
    "resp_rate": ["ademhalingsfrequentie", "resp_rate", "respiratoryrate", "rr"],
    "temperature": ["temperatuur", "temperature", "temp"],
    "wbc": ["leukocyten", "wbc", "whitebloodcell"],
    "bun": ["ureum", "bun", "bloodureanitrogen"],
}


CARE_CONTEXT_ALIASES = {
    "er": {
        "heart_rate": ["edheartrate", "heartrate", "hr"],
        "resp_rate": ["edrespiratoryrate", "respiratoryrate", "rr"],
        "temperature": ["edtemperature", "temperature", "temp"],
        "wbc": ["edwbc", "wbc", "whitebloodcell"],
        "bun": ["edbun", "bun", "bloodureanitrogen"],
    },
    "internal": {
        "heart_rate": ["wardheartrate", "heartrate", "hr"],
        "resp_rate": ["wardrespiratoryrate", "respiratoryrate", "rr"],
        "temperature": ["wardtemperature", "temperature", "temp"],
        "wbc": ["wardwbc", "wbc", "whitebloodcell"],
        "bun": ["wardbun", "bun", "bloodureanitrogen"],
    },
}


def available_massive_datasets() -> list[dict[str, str]]:
    """Large-scale healthcare datasets for advanced research planning."""
    return [
        {"name": "MIMIC-IV", "url": "https://www.kaggle.com/datasets/hussameldinanwer/mimic-iii", "type": "massive_ehr"},
        {"name": "eICU-CRD 2.0", "url": "https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k", "type": "massive_ehr"},
        {"name": "MIMIC-III v1.4", "url": "https://www.kaggle.com/datasets/hussameldinanwer/mimic-iii", "type": "massive_ehr"},
        {"name": "AmsterdamUMCdb", "url": AMSTERDAMUMCDB_URL, "type": "massive_ehr"},
        {"name": "HiRID", "url": "https://huggingface.co/datasets?search=icu", "type": "massive_ehr"},
        {"name": "MIMIC-CXR-JPG", "url": "https://stanfordmlgroup.github.io/competitions/chexpert/", "type": "massive_imaging"},
        {"name": "CheXpert", "url": "https://stanfordmlgroup.github.io/competitions/chexpert/", "type": "massive_imaging"},
        {"name": "UK Biobank", "url": "https://www.ukbiobank.ac.uk/", "type": "biobank"},
        {"name": "All of Us", "url": "https://www.researchallofus.org/", "type": "biobank"},
    ]


def available_open_ehr_datasets() -> list[dict[str, str]]:
    """Curated open/open-access medical record datasets."""
    return [
        {"name": "MIMIC-IV", "url": "https://www.kaggle.com/datasets/hussameldinanwer/mimic-iii", "type": "ehr"},
        {"name": "eICU-CRD 2.0", "url": "https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k", "type": "ehr"},
        {"name": "MIMIC-III v1.4", "url": "https://www.kaggle.com/datasets/hussameldinanwer/mimic-iii", "type": "ehr"},
        {"name": "Synthea downloads", "url": "https://synthea.mitre.org/downloads", "type": "synthetic_ehr"},
        {"name": "LHS synthetic-data", "url": "https://github.com/lhs-open/synthetic-data", "type": "synthetic_ehr"},
    ]


def available_hf_like_datasets() -> list[dict[str, str]]:
    """Hugging Face / similar sources for quick experimentation."""
    return [
        {"name": "HF ICU sepsis prediction", "url": "https://huggingface.co/datasets/Abdu347/icu-sepsis-prediction", "type": "huggingface"},
        {"name": "HF ICU search", "url": "https://huggingface.co/datasets?search=icu", "type": "huggingface"},
    ]


def available_kaggle_like_datasets() -> list[dict[str, str]]:
    """Kaggle and similar sources for quick experimentation."""
    return [
        {"name": "Kaggle MIMIC-III-10k", "url": "https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k", "type": "kaggle"},
        {"name": "Kaggle MIMIC III mirror", "url": "https://www.kaggle.com/datasets/hussameldinanwer/mimic-iii", "type": "kaggle"},
        {"name": "Kaggle ICU mortality", "url": "https://www.kaggle.com/datasets/fdemoribajolin/death-classification-icu", "type": "kaggle"},
        {"name": "HuggingFace ICU sepsis", "url": "https://huggingface.co/Abdu347/icu-sepsis-prediction", "type": "similar"},
    ]


def available_open_datasets() -> list[dict[str, str]]:
    """Catalog of non-demo datasets relevant to this project."""
    return [
        {"name": "eICU-CRD 2.0 (Full, credentialed)", "url": "https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k", "type": "icu_tabular"},
        {"name": "MIMIC-IV Waveform DB 0.1.0", "url": "https://huggingface.co/datasets?search=ecg", "type": "waveform"},
    ]

        {"name": "eICU-CRD 2.0 (Full, credentialed)", "url": "https://physionet.org/content/eicu-crd/2.0/", "type": "icu_tabular"},
        {"name": "MIMIC-IV Waveform DB 0.1.0", "url": "https://physionet.org/content/mimic4wdb/0.1.0/", "type": "waveform"},
    ]

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


def convert_eicu_to_clinical(input_csv: Path, output_csv: Path, care_context: str = "icu") -> list[dict[str, str]]:
def convert_eicu_to_clinical(input_csv: Path, output_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    alias_map = {k: list(v) for k, v in EICU_ALIASES.items()}
    if care_context in CARE_CONTEXT_ALIASES:
        for k, v in CARE_CONTEXT_ALIASES[care_context].items():
            alias_map[k] = list(v)
    source_map = {target: _find_column(columns, aliases) for target, aliases in alias_map.items()}
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




def convert_amsterdamumc_to_mimic_format(input_csv: Path, output_csv: Path) -> list[dict[str, str]]:
    """Convert AmsterdamUMCdb-style CSV columns into MIMIC III/IV-like clinical columns."""
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    source_map = {target: _find_column(columns, aliases) for target, aliases in AMSTERDAM_TO_MIMIC_ALIASES.items()}

    out_rows: list[dict[str, str]] = []
    for row in rows:
        out = {}
        for c in CLINICAL_COLUMNS:
            src = source_map.get(c)
            val = "" if src is None else row.get(src, "")
            if c == "age" and isinstance(val, str) and "-" in val:
                # agegroup like "60-69" -> midpoint 65
                try:
                    a, b = val.split("-", 1)
                    val = str((int(a) + int(b)) // 2)
                except Exception:
                    pass
            out[c] = val
        out_rows.append(out)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLINICAL_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    return out_rows

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert eICU table to clinical.csv schema")
    parser.add_argument("--input", "-i", type=Path, default=None, help="Path to eICU CSV file")
    parser.add_argument("--output", "-o", type=Path, default=Path("clinical_eicu.csv"), help="Output CSV")
    parser.add_argument("--autofetch", action="store_true", help="Auto-download non-PhysioNet source archive if --input is not set")
    parser.add_argument("--fetch_url", type=str, default=EICU_FULL_PATIENT_CSV_GZ, help="Source URL used with --autofetch (non-PhysioNet)")
    parser.add_argument("--fetched_input", type=Path, default=Path("eicu_patient_autofetch.csv"), help="Where to store downloaded source CSV")
    parser.add_argument("--care_context", choices=["icu", "er", "internal"], default="icu", help="Column mapping context: ICU default, or ER / internal-medicine variants")
    parser.add_argument("--list_open_datasets", action="store_true", help="Print open/public dataset catalog and exit")
    parser.add_argument("--list_kaggle_like", action="store_true", help="Print Kaggle/similar dataset sources and exit")
    parser.add_argument("--list_hf_like", action="store_true", help="Print HuggingFace/similar dataset sources and exit")
    parser.add_argument("--list_open_ehr", action="store_true", help="Print curated open/open-access EHR dataset list and exit")
    parser.add_argument("--list_massive", action="store_true", help="Print extended massive healthcare dataset list and exit")
    parser.add_argument("--amsterdam_to_mimic", action="store_true", help="Convert AmsterdamUMCdb-style CSV to MIMIC III/IV-like clinical schema")
    parser.add_argument("--autofetch", action="store_true", help="Auto-download full eICU patient table if --input is not set")
    parser.add_argument("--fetch_url", type=str, default=EICU_FULL_PATIENT_CSV_GZ, help="Source URL used with --autofetch (non-demo only)")
    parser.add_argument("--fetched_input", type=Path, default=Path("eicu_patient_autofetch.csv"), help="Where to store downloaded source CSV")
    parser.add_argument("--list_open_datasets", action="store_true", help="Print open/public dataset catalog and exit")
    parser.add_argument("--list_kaggle_like", action="store_true", help="Print Kaggle/similar dataset sources and exit")
    parser.add_argument("--list_hf_like", action="store_true", help="Print HuggingFace/similar dataset sources and exit")
    parser.add_argument("--autofetch", action="store_true", help="Auto-download eICU demo patient table if --input is not set")
    parser.add_argument("--fetch_url", type=str, default=EICU_DEMO_PATIENT_CSV_GZ, help="Source URL used with --autofetch")
    parser.add_argument("--fetched_input", type=Path, default=Path("eicu_patient_autofetch.csv"), help="Where to store downloaded source CSV")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to eICU CSV file")
    parser.add_argument("--output", "-o", type=Path, default=Path("clinical_eicu.csv"), help="Output CSV")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_open_datasets:
        for item in available_open_datasets():
            print(f"{item['name']} ({item['type']}): {item['url']}")
        return

    if args.list_massive:
        for item in available_massive_datasets():
            print(f"{item['name']} ({item['type']}): {item['url']}")
        return

    if args.list_open_ehr:
        for item in available_open_ehr_datasets():
            print(f"{item['name']} ({item['type']}): {item['url']}")
        return

    if args.list_hf_like:
        for item in available_hf_like_datasets():
            print(f"{item['name']} ({item['type']}): {item['url']}")
        return

    if args.list_kaggle_like:
        for item in available_kaggle_like_datasets():
            print(f"{item['name']} ({item['type']}): {item['url']}")
        return

    input_csv = args.input
    if input_csv is None:
        if not args.autofetch:
            raise ValueError("Provide --input, or use --autofetch to download full eICU data")
        if "demo" in args.fetch_url.lower():
            raise ValueError("Demo/PhysioNet URLs are disabled for this workflow.")
        input_csv = fetch_eicu_csv(args.fetch_url, args.fetched_input)
        print(f"[INFO] Downloaded eICU source -> {input_csv}")

    if args.amsterdam_to_mimic:
        converted = convert_amsterdamumc_to_mimic_format(input_csv, args.output)
    else:
        converted = convert_eicu_to_clinical(input_csv, args.output, care_context=args.care_context)
            raise ValueError("Demo sources are disabled. Use full non-demo eICU source URL.")
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
