"""CLI to generate pathogen predictions from clinical vitals.

Supports:
 - Single prediction via `--single` flags: --temperature_c --wbc --spo2 --age
 - Batch prediction from CSV via `--input` (expects columns: temperature_c,wbc,spo2,age)

Outputs a CSV or JSON file with class probabilities by default.

Examples:
  python -m scripts.predict_pathogen --single --task specpath --temperature_c 37.1 --wbc 7000 --spo2 96 --age 60 --format csv --output out.csv
  python -m scripts.predict_pathogen --input patients.csv --format json --output preds.json
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import List
import numpy as np

import csv
import tensorflow as tf

SCALER_FILE = Path("clin_scaler.npz")
ENCODER_FILE = Path("clin_encoder.keras")
HEAD_FILE = Path("clin_head.keras")
PATHOGEN_VOCAB = Path("pathogen_vocab.json")


def _class_names(task: str) -> List[str]:
    if task == "binary":
        return ["Normal", "Pneumonia"]
    if task == "pathogen":
        return ["Bacterial", "Viral"]
    if task == "3class":
        return ["Normal", "Bacterial", "Viral"]
    if task == "specpath":
        if not PATHOGEN_VOCAB.exists():
            raise FileNotFoundError(f"Missing {PATHOGEN_VOCAB}. Train/build features first.")
        return json.loads(PATHOGEN_VOCAB.read_text())["classes"]
    raise ValueError(f"Unknown task: {task}")


def _load_models():
    if not ENCODER_FILE.exists() or not HEAD_FILE.exists():
        raise FileNotFoundError("Missing clin_encoder.keras / clin_head.keras. Train first.")
    if not SCALER_FILE.exists():
        raise FileNotFoundError(f"Missing {SCALER_FILE}. Train first (it is saved during training).")

    scaler = np.load(SCALER_FILE)
    mu = scaler["mu"].astype(np.float32)
    sd = scaler["sd"].astype(np.float32)

    clin_enc = tf.keras.models.load_model(str(ENCODER_FILE))
    clin_head = tf.keras.models.load_model(str(HEAD_FILE))

    return mu, sd, clin_enc, clin_head


def _predict_array(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, clin_enc, clin_head, task: str):
    # x: (n, 4)
    x = (x - mu) / (sd + 1e-6)
    z = clin_enc(x, training=False)
    out = clin_head(z, training=False)
    # Some Keras models return EagerTensor, others return numpy -- normalize
    try:
        out = out.numpy()
    except Exception:
        out = np.asarray(out)

    if out.ndim == 1:
        out = out.reshape(1, -1)

    names = _class_names(task)
    preds = []
    for row in out:
        if len(names) == 2:
            # binary sigmoid
            p1 = float(row[0])
            probs = [1.0 - p1, p1]
        else:
            probs = row.tolist()
        preds.append(dict(zip(names, probs)))
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict pathogen/class probabilities from vitals")
    ap.add_argument("--task", choices=["binary", "3class", "pathogen", "specpath"], default="specpath")
    ap.add_argument("--input", type=str, default=None, help="CSV input file (columns: temperature_c,wbc,spo2,age)")
    ap.add_argument("--single", action="store_true", help="Single prediction via flags")
    ap.add_argument("--temperature_c", type=float, default=None)
    ap.add_argument("--wbc", type=float, default=None)
    ap.add_argument("--spo2", type=float, default=None)
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--output", "-o", type=str, default=None, help="Output file (CSV or JSON based on --format)")
    ap.add_argument("--format", choices=["csv", "json"], default="csv")
    ap.add_argument("--top", type=int, default=None, help="If provided, include only top-K classes per row (useful for 'specpath')")
    args = ap.parse_args(argv)

    mu, sd, clin_enc, clin_head = _load_models()

    if args.single:
        for nm in ("temperature_c", "wbc", "spo2", "age"):
            if getattr(args, nm) is None:
                raise ValueError(f"--single requires --{nm}")
        x = np.array([[args.temperature_c, args.wbc, args.spo2, args.age]], dtype=np.float32)
        preds = _predict_array(x, mu, sd, clin_enc, clin_head, args.task)
        row = preds[0]
        if args.top:
            items = sorted(row.items(), key=lambda t: t[1], reverse=True)[: args.top]
            row = dict(items)
        # Print and optionally save
        print(json.dumps(row, indent=2))
        if args.output:
            outp = Path(args.output)
            if args.format == "csv":
                if pd is None:
                    raise RuntimeError("Pandas is required for CSV output. Install pandas and retry.")
                pd.DataFrame([row]).to_csv(outp, index=False)
            else:
                outp.write_text(json.dumps(row), encoding="utf-8")
        return

    if args.input is None:
        raise ValueError("Either --input or --single must be provided")

    # Read CSV input using stdlib csv (avoid pandas dependency)
    required = ["temperature_c", "wbc", "spo2", "age"]
    rows = []
    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for c in required:
            if c not in header:
                raise ValueError(f"Input CSV must contain column: {c}")
        for r in reader:
            rows.append({c: float(r[c]) for c in required})

    import numpy as _np
    X = _np.array([[r[c] for c in required] for r in rows], dtype=_np.float32)
    preds = _predict_array(X, mu, sd, clin_enc, clin_head, args.task)

    # Expand predictions and include original columns
    out_rows = []
    class_cols = _class_names(args.task)
    for i, p in enumerate(preds):
        row = {k: float(v) for k, v in p.items()}
        for c in required:
            row[c] = float(rows[i][c])
        out_rows.append(row)

    # Top-K handling
    if args.top and args.top < len(class_cols):
        tops = []
        for r in out_rows:
            items = sorted([(c, r[c]) for c in class_cols], key=lambda t: t[1], reverse=True)[: args.top]
            tops.append({"top": items, **{c: r[c] for c in required}})
        output_data = tops
    else:
        output_data = out_rows

    # Write output
    if args.output:
        outp = Path(args.output)
        if args.format == "csv":
            # write CSV with class columns + required columns
            cols = class_cols + required
            with open(outp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in output_data:
                    # ensure keys for all cols
                    w.writerow({k: r.get(k, "") for k in cols})
        else:
            outp.write_text(json.dumps(output_data), encoding="utf-8")
    else:
        # Print a short preview
        print(json.dumps(output_data[:5], indent=2))


if __name__ == "__main__":
    main()
