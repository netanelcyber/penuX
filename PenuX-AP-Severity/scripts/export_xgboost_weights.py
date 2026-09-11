"""Export normalized XGBoost feature-importance weights from development data.

The exported values are native XGBoost *gain* importances, normalized to sum to
1.0. They are explanatory feature weights, NOT linear prediction coefficients.

RESEARCH USE ONLY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", default="Diagnostic Result")
    p.add_argument(
        "--positive-raw-value",
        type=int,
        choices=[0, 1],
        default=1,
        help="Raw source value that represents SAP. Output labels are normalized to 1=SAP.",
    )
    p.add_argument("--output", default="docs/model-weights.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from xgboost import XGBClassifier

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    raw_y = pd.to_numeric(df[args.target], errors="coerce")
    keep = raw_y.isin([0, 1])
    X = df.loc[keep].drop(columns=[args.target]).copy()
    raw_y = raw_y.loc[keep].astype(int)

    # Normalize outcome semantics explicitly: 1 always means SAP in this script.
    y = (raw_y == args.positive_raw_value).astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.seed
    )

    imputer = SimpleImputer(strategy="median")
    X_dev_i = imputer.fit_transform(X_dev)

    n_pos = int((y_dev == 1).sum())
    n_neg = int((y_dev == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=2,
        verbosity=0,
    )
    model.fit(X_dev_i, y_dev)

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    split_count = booster.get_score(importance_type="weight")

    feature_names = list(X.columns)
    gain_by_name: dict[str, float] = {}
    count_by_name: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        gain_by_name[name] = float(gain.get(f"f{i}", 0.0))
        count_by_name[name] = float(split_count.get(f"f{i}", 0.0))

    total_gain = float(sum(gain_by_name.values()))
    if total_gain <= 0:
        raise RuntimeError("XGBoost returned zero total gain")

    rows = []
    for name in feature_names:
        normalized = gain_by_name[name] / total_gain
        rows.append(
            {
                "feature": name,
                "gain": gain_by_name[name],
                "gain_weight": normalized,
                "gain_percent": 100.0 * normalized,
                "split_count": count_by_name[name],
            }
        )
    rows.sort(key=lambda r: r["gain_weight"], reverse=True)

    payload = {
        "research_use_only": True,
        "importance_type": "xgboost_gain",
        "important_note": "Gain weights are explanatory importances, not linear coefficients and cannot be multiplied by raw laboratory values to reproduce XGBoost predictions.",
        "dataset": Path(args.data).name,
        "target": args.target,
        "raw_value_representing_SAP": int(args.positive_raw_value),
        "normalized_target": "0=non-SAP, 1=SAP",
        "n_total": int(len(X)),
        "n_development": int(len(X_dev)),
        "n_held_out_not_used_for_weights": int(len(X_test)),
        "development_SAP_rate": float(y_dev.mean()),
        "scale_pos_weight": float(scale_pos_weight),
        "model_parameters": model.get_params(),
        "features": rows,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"top_15": rows[:15], "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
