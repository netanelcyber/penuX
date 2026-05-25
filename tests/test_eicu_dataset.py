from pathlib import Path
import csv

from scripts.eicu_dataset import convert_eicu_to_clinical


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
