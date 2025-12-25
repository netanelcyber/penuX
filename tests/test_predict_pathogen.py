import sys
from pathlib import Path
import json
import numpy as np
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import predict_pathogen


class DummyModel:
    def __init__(self, out):
        self._out = np.array(out, dtype=np.float32)

    def __call__(self, x, training=False):
        # ignore input and return fixed logits / prob vector
        n = x.shape[0] if hasattr(x, "shape") else 1
        return np.tile(self._out.reshape(1, -1), (n, 1))


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def test_single_prediction_csv(tmp_path, monkeypatch):
    # Create dummy scaler
    mu = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sd = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    np.savez(tmp_path / "clin_scaler.npz", mu=mu, sd=sd)

    # Touch model files
    (tmp_path / "clin_encoder.keras").write_text("")
    (tmp_path / "clin_head.keras").write_text("")

    # monkeypatch global paths to tmp
    monkeypatch.setattr(predict_pathogen, "SCALER_FILE", tmp_path / "clin_scaler.npz")
    monkeypatch.setattr(predict_pathogen, "ENCODER_FILE", tmp_path / "clin_encoder.keras")
    monkeypatch.setattr(predict_pathogen, "HEAD_FILE", tmp_path / "clin_head.keras")

    # Dummy models: head returns softmax probs for 2 classes (specpath uses many classes, but for test keep 3class)
    def fake_load(p):
        if str(p).endswith("clin_encoder.keras"):
            return DummyModel([0.1, 0.2])
        return DummyModel([0.3, 0.7])

    monkeypatch.setattr(predict_pathogen.tf.keras.models, "load_model", fake_load)

    # Run single prediction and save CSV
    out = tmp_path / "out.csv"
    args = [
        "--single",
        "--task",
        "3class",
        "--temperature_c",
        "36.6",
        "--wbc",
        "7000",
        "--spo2",
        "96",
        "--age",
        "60",
        "--format",
        "csv",
        "--output",
        str(out),
    ]
    predict_pathogen.main(args)
    rows = read_csv(out)
    assert len(rows) == 1


def test_batch_prediction_json(tmp_path, monkeypatch):
    mu = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sd = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    np.savez(tmp_path / "clin_scaler.npz", mu=mu, sd=sd)
    (tmp_path / "clin_encoder.keras").write_text("")
    (tmp_path / "clin_head.keras").write_text("")

    monkeypatch.setattr(predict_pathogen, "SCALER_FILE", tmp_path / "clin_scaler.npz")
    monkeypatch.setattr(predict_pathogen, "ENCODER_FILE", tmp_path / "clin_encoder.keras")
    monkeypatch.setattr(predict_pathogen, "HEAD_FILE", tmp_path / "clin_head.keras")

    def fake_load(p):
        if str(p).endswith("clin_encoder.keras"):
            return DummyModel([0.1, 0.2, 0.7])
        return DummyModel([0.1, 0.2, 0.7])

    monkeypatch.setattr(predict_pathogen.tf.keras.models, "load_model", fake_load)

    # Prepare input CSV
    rows = [
        {"temperature_c": 36.6, "wbc": 7000, "spo2": 96, "age": 60},
        {"temperature_c": 37.0, "wbc": 6000, "spo2": 95, "age": 50},
    ]
    csv_in = tmp_path / "in.csv"
    write_csv(csv_in, rows, ["temperature_c", "wbc", "spo2", "age"])

    out = tmp_path / "out.json"
    args = ["--input", str(csv_in), "--format", "json", "--output", str(out), "--task", "3class"]
    predict_pathogen.main(args)

    txt = out.read_text()
    arr = json.loads(txt)
    assert isinstance(arr, list)
    assert len(arr) == 2


def test_single_threshold_round_json(tmp_path, monkeypatch):
    mu = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sd = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    np.savez(tmp_path / "clin_scaler.npz", mu=mu, sd=sd)
    (tmp_path / "clin_encoder.keras").write_text("")
    (tmp_path / "clin_head.keras").write_text("")

    monkeypatch.setattr(predict_pathogen, "SCALER_FILE", tmp_path / "clin_scaler.npz")
    monkeypatch.setattr(predict_pathogen, "ENCODER_FILE", tmp_path / "clin_encoder.keras")
    monkeypatch.setattr(predict_pathogen, "HEAD_FILE", tmp_path / "clin_head.keras")

    def fake_load(p):
        if str(p).endswith("clin_encoder.keras"):
            return DummyModel([0.1, 0.6, 0.3])
        return DummyModel([0.1, 0.6, 0.3])

    monkeypatch.setattr(predict_pathogen.tf.keras.models, "load_model", fake_load)

    out = tmp_path / "out.json"
    args = [
        "--single",
        "--task",
        "3class",
        "--temperature_c",
        "36.6",
        "--wbc",
        "7000",
        "--spo2",
        "96",
        "--age",
        "60",
        "--format",
        "json",
        "--output",
        str(out),
        "--threshold",
        "0.5",
        "--round",
        "2",
    ]
    predict_pathogen.main(args)
    txt = out.read_text()
    data = json.loads(txt)
    # Only the class with prob 0.6 should remain (rounded to 2 decimals)
    assert isinstance(data, dict)
    assert len(data.keys()) == 1
    assert list(data.values())[0] == 0.6


def test_batch_keep_cols_and_sep_and_round(tmp_path, monkeypatch):
    mu = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sd = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    np.savez(tmp_path / "clin_scaler.npz", mu=mu, sd=sd)
    (tmp_path / "clin_encoder.keras").write_text("")
    (tmp_path / "clin_head.keras").write_text("")

    monkeypatch.setattr(predict_pathogen, "SCALER_FILE", tmp_path / "clin_scaler.npz")
    monkeypatch.setattr(predict_pathogen, "ENCODER_FILE", tmp_path / "clin_encoder.keras")
    monkeypatch.setattr(predict_pathogen, "HEAD_FILE", tmp_path / "clin_head.keras")

    def fake_load(p):
        if str(p).endswith("clin_encoder.keras"):
            return DummyModel([0.12, 0.88])
        return DummyModel([0.12, 0.88])

    monkeypatch.setattr(predict_pathogen.tf.keras.models, "load_model", fake_load)

    # Prepare TSV input with extra patient_id column
    rows = [
        {"patient_id": "a1", "temperature_c": 36.6, "wbc": 7000, "spo2": 96, "age": 60},
        {"patient_id": "b2", "temperature_c": 37.0, "wbc": 6000, "spo2": 95, "age": 50},
    ]
    csv_in = tmp_path / "in.tsv"
    with open(csv_in, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "temperature_c", "wbc", "spo2", "age"], delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out = tmp_path / "out.json"
    args = [
        "--input",
        str(csv_in),
        "--format",
        "json",
        "--output",
        str(out),
        "--task",
        "binary",
        "--keep_cols",
        "patient_id",
        "--sep",
        "\t",
        "--round",
        "2",
    ]
    predict_pathogen.main(args)

    txt = out.read_text()
    arr = json.loads(txt)
    assert isinstance(arr, list)
    assert len(arr) == 2
    assert all("patient_id" in r for r in arr)
    # probabilities should be rounded to 2 decimals
    for r in arr:
        vals = [v for k, v in r.items() if k in ["Normal", "Pneumonia"]]
        assert all(isinstance(v, float) for v in vals)
        assert all(round(v, 2) == v for v in vals)
