from pathlib import Path
import csv
import gzip

from scripts.eicu_dataset import convert_eicu_to_clinical, fetch_eicu_csv


def test_convert_eicu_to_clinical(tmp_path: Path):
    src = tmp_path / "eicu_sample.csv"
    dst = tmp_path / "clinical_eicu.csv"

    with src.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["age", "heartrate", "respiratoryrate", "temperature", "wbc", "bun"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"age": 65, "heartrate": 88, "respiratoryrate": 20, "temperature": 37.2, "wbc": 11.2, "bun": 18},
                {"age": 42, "heartrate": 72, "respiratoryrate": 16, "temperature": 36.8, "wbc": 7.5, "bun": 12},
            ]
        )

    out = convert_eicu_to_clinical(src, dst)

    assert dst.exists()
    assert list(out[0].keys()) == ["age", "heart_rate", "resp_rate", "temperature", "wbc", "bun"]
    assert out[0]["heart_rate"] == "88"


def test_fetch_eicu_csv_from_gzip_file_url(tmp_path: Path):
    raw = tmp_path / "patient.csv"
    raw.write_text("age,heartrate\n50,77\n", encoding="utf-8")

    gz_path = tmp_path / "patient.csv.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(raw.read_bytes())

    out = tmp_path / "downloaded.csv"
    fetched = fetch_eicu_csv(gz_path.resolve().as_uri(), out)

    assert fetched == out
    assert out.read_text(encoding="utf-8") == "age,heartrate\n50,77\n"
