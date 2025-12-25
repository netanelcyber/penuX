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
        # ensure row is a 1-D array
        r = np.asarray(row).reshape(-1)
        if len(names) == 2:
            # handle binary outputs: either a single-sigmoid value or a 2-class vector
            if r.shape[0] == 1:
                p1 = float(r[0])
                probs = [1.0 - p1, p1]
            elif r.shape[0] == 2:
                vals = r.tolist()
                s = sum(vals)
                if s > 0:
                    probs = [float(v) / float(s) for v in vals]
                else:
                    probs = [float(vals[0]), float(vals[1])]
            else:
                # fallback: take first value as sigmoid
                p1 = float(r[0])
                probs = [1.0 - p1, p1]
        else:
            probs = r.tolist()
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

    # New convenience parameters
    ap.add_argument("--threshold", type=float, default=None, help="Only include class probs >= threshold (0..1), per-row")
    ap.add_argument("--keep_cols", type=str, default=None, help="Comma-separated list of extra input columns to include in output rows")
    ap.add_argument("--sep", type=str, default="," , help='Input CSV separator/delimiter (default=",")')
    ap.add_argument("--round", type=int, default=None, help="Round probabilities to N decimals in output")
    ap.add_argument("--no_print", action="store_true", help="Suppress printing to stdout; write only to --output if provided")

    # Output ordering / sorting
    ap.add_argument("--order_classes", action="store_true", help="Order class probabilities per row (descending) in outputs")
    ap.add_argument("--sort_rows_by", type=str, default=None, help="Sort batch rows by given key: 'top_prob' (or class name) or input column name")
    ap.add_argument("--sort_desc", action="store_true", help="Sort rows in descending order (default for probabilities)")

    args = ap.parse_args(argv)

    mu, sd, clin_enc, clin_head = _load_models()

    def _process_row_values(d: dict, class_cols: list) -> dict:
        """Apply threshold/rounding for output formatting. Returns a new dict."""
        out = {}
        # Handle class columns
        for c in class_cols:
            v = d.get(c)
            if v is None:
                continue
            if args.threshold is not None:
                if args.format == "json":
                    # JSON: omit classes below threshold
                    if v < args.threshold:
                        continue
                else:
                    # CSV: set below-threshold to empty string for readability
                    if v < args.threshold:
                        out[c] = ""
                        continue
            if args.round is not None and isinstance(v, float):
                out[c] = round(float(v), args.round)
            else:
                out[c] = float(v)
        return out

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

        class_cols = list(row.keys())
        processed = _process_row_values(row, class_cols)

        # Print (unless suppressed) and optionally save
        if not args.no_print:
            if args.format == "json":
                print(json.dumps(processed, indent=2))
            else:
                print(processed)

        if args.output:
            outp = Path(args.output)
            if args.format == "csv":
                cols = class_cols
                if args.order_classes:
                    cols = ["preds"] + cols
                with open(outp, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    # build row dict, serializing preds if necessary
                    rowdict = {}
                    for k in cols:
                        if k == "preds":
                            # create ordered preds list
                            items = sorted(row.items(), key=lambda t: t[1], reverse=True)
                            if args.round is not None:
                                items = [(n, round(float(v), args.round)) for n, v in items]
                            rowdict["preds"] = ";".join(f"{n}:{v}" for n, v in items)
                        else:
                            rowdict[k] = processed.get(k, "")
                    w.writerow(rowdict)
            else:
                outp.write_text(json.dumps(processed), encoding="utf-8")
        return

    if args.input is None:
        raise ValueError("Either --input or --single must be provided")

    # Read CSV input using stdlib csv (avoid pandas dependency)
    required = ["temperature_c", "wbc", "spo2", "age"]
    rows = []
    original_rows = []
    keep_cols = []
    if args.keep_cols:
        keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]

    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=args.sep)
        header = reader.fieldnames or []
        for c in required:
            if c not in header:
                raise ValueError(f"Input CSV must contain column: {c}")
        for kc in keep_cols:
            if kc not in header:
                raise ValueError(f"Requested --keep_cols column not in input CSV: {kc}")
        for r in reader:
            original_rows.append(r)
            rows.append({c: float(r[c]) for c in required})

    import numpy as _np
    X = _np.array([[r[c] for c in required] for r in rows], dtype=_np.float32)
    preds = _predict_array(X, mu, sd, clin_enc, clin_head, args.task)

    # Expand predictions and include original/keep columns
    out_rows = []
    class_cols = _class_names(args.task)
    # We'll also keep the raw class dicts to compute sort keys
    raw_class_dicts = []
    for i, p in enumerate(preds):
        raw = {k: float(v) for k, v in p.items()}
        raw_class_dicts.append(raw)
        row = {c: float(rows[i][c]) for c in required}
        for kc in keep_cols:
            row[kc] = original_rows[i].get(kc, "")
        # Apply rounding and threshold for class columns (for CSV/individual columns)
        class_proc = _process_row_values({**raw, **row}, class_cols)
        # merge processed class cols with the rest
        merged = {**class_proc, **row}
        out_rows.append(merged)

    # If requested, produce ordered class lists per row
    if args.order_classes:
        # For JSON: add 'preds' as ordered list of [class, prob]
        # For CSV: we'll also add a 'preds' column as semicolon-separated 'Class:prob' strings
        preds_col = []
        for raw in raw_class_dicts:
            items = sorted(raw.items(), key=lambda t: t[1], reverse=True)
            if args.round is not None:
                items = [(n, round(float(v), args.round)) for n, v in items]
            else:
                items = [(n, float(v)) for n, v in items]
            preds_col.append(items)
        # attach preds_col to out_rows (as structured list for JSON and string for CSV)
        for i, items in enumerate(preds_col):
            out_rows[i]["preds"] = items

    # Row sorting
    if args.sort_rows_by is not None:
        key = args.sort_rows_by
        def sort_key_fn(r, raw):
            # raw: raw_class_dicts counterpart
            if key == "top_prob":
                return max(raw.values()) if len(raw) > 0 else -float("inf")
            if key in raw:
                return raw.get(key)
            # fall back to using provided column from row
            return r.get(key)

        combined = list(zip(out_rows, raw_class_dicts))
        # default descending for probs/class names unless sort_desc explicitly set false
        reverse = True if (args.sort_desc or key in (["top_prob"] + class_cols)) else False
        combined.sort(key=lambda t: (sort_key_fn(t[0], t[1]) is None, sort_key_fn(t[0], t[1])), reverse=reverse)
        out_rows = [t[0] for t in combined]

    # Top-K handling
    if args.top and args.top < len(class_cols):
        tops = []
        for r, orig, raw in zip(out_rows, original_rows, raw_class_dicts):
            items = sorted([(c, raw.get(c)) for c in class_cols if c in raw], key=lambda t: (t[1] if t[1] is not None else -1), reverse=True)[: args.top]
            tops.append({"top": items, **{c: r[c] for c in required}, **{kc: orig.get(kc, "") for kc in keep_cols}})
        output_data = tops
    else:
        output_data = out_rows

    # Write output
    if args.output:
        outp = Path(args.output)
        if args.format == "csv":
            # write CSV with class columns + required columns + keep cols
            cols = class_cols + required + keep_cols
            if args.order_classes:
                cols = ["preds"] + cols
            with open(outp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in output_data:
                    # ensure keys for all cols, serialize preds if present
                    rowdict = {}
                    for k in cols:
                        if k == "preds":
                            preds_val = r.get("preds")
                            if isinstance(preds_val, list):
                                # already ordered list of tuples
                                if args.round is not None:
                                    rowdict["preds"] = ";".join(f"{n}:{round(float(v), args.round)}" for n, v in preds_val)
                                else:
                                    rowdict["preds"] = ";".join(f"{n}:{v}" for n, v in preds_val)
                            else:
                                rowdict["preds"] = preds_val if preds_val is not None else ""
                        else:
                            rowdict[k] = r.get(k, "")
                    w.writerow(rowdict)
        elif args.format == "json":
            # JSON dump may fail on numpy types; ensure conversion to native types
            def _convert(o):
                if isinstance(o, list):
                    return [_convert(x) for x in o]
                if isinstance(o, dict):
                    return {str(k): _convert(v) for k, v in o.items()}
                try:
                    # numpy scalar
                    import numpy as _np
                    if isinstance(o, _np.generic):
                        return o.item()
                except Exception:
                    pass
                return o

            safe = _convert(output_data)
            outp.write_text(json.dumps(safe), encoding="utf-8")
        else:
            # fallback
            outp.write_text(json.dumps(output_data), encoding="utf-8")
    else:
        if not args.no_print:
            # Print a short preview
            print(json.dumps(output_data[:5], indent=2))

if __name__ == "__main__":
    main()
