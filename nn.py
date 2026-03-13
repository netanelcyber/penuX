#!/usr/bin/env python3
"""
PenuX unified kidney-aware neural pipeline in one Python file.

What this file does:
- accepts either a prepared CSV (--csv) or raw MIMIC-III files (--chartevents + --d-items)
- engineers kidney features in the same file
- supports DNN, RNN, and LSTM from one entrypoint
- exports prepared datasets, metrics, predictions, and model weights

Typical usage:
    python penux_unified_nn.py --csv kidney_function_data.csv --target pathogen --model dnn

    python penux_unified_nn.py \
      --chartevents dataset/mimic/mimic-iii-clinical-database-demo-1.4/CHARTEVENTS.csv \
      --d-items dataset/mimic/mimic-iii-clinical-database-demo-1.4/D_ITEMS.csv \
      --target aki_stage_label --model dnn

    python penux_unified_nn.py \
      --chartevents dataset/mimic/mimic-iii-clinical-database-demo-1.4/CHARTEVENTS.csv \
      --d-items dataset/mimic/mimic-iii-clinical-database-demo-1.4/D_ITEMS.csv \
      --target severe_aki_label --model lstm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Reproducibility
# -----------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Column name helpers
# -----------------------------


def normalize_name(name: str) -> str:
    s = str(name).replace("\ufeff", "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_column_map(df: pd.DataFrame) -> Dict[str, str]:
    return {normalize_name(c): c for c in df.columns}


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cmap = build_column_map(df)
    for c in candidates:
        key = normalize_name(c)
        if key in cmap:
            return cmap[key]
    return None


CANONICAL_COMMON_COLS = {
    "itemid": "ITEMID",
    "subject_id": "SUBJECT_ID",
    "hadm_id": "HADM_ID",
    "icustay_id": "ICUSTAY_ID",
    "charttime": "CHARTTIME",
    "storetime": "STORETIME",
    "valuenum": "VALUENUM",
    "value": "VALUE",
    "valueuom": "VALUEUOM",
    "label": "LABEL",
    "abbreviation": "ABBREVIATION",
    "category": "CATEGORY",
    "param_type": "PARAM_TYPE",
    "linksto": "LINKSTO",
    "unitname": "UNITNAME",
    "dbsource": "DBSOURCE",
}


def canonicalize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for c in df.columns:
        key = normalize_name(c)
        if key in CANONICAL_COMMON_COLS:
            rename_map[c] = CANONICAL_COMMON_COLS[key]
    out = df.rename(columns=rename_map).copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


# -----------------------------
# CSV reading helpers
# -----------------------------


def read_csv_selected(path: str, wanted_cols: Optional[Iterable[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Read a CSV robustly while handling BOM/whitespace/case differences in headers.
    wanted_cols should be expressed in normalized form, e.g. ['itemid', 'charttime'].
    """
    header = pd.read_csv(path, nrows=0)
    original_cols = list(header.columns)
    norm_to_original = {normalize_name(c): c for c in original_cols}

    if wanted_cols is None:
        usecols = None
    else:
        usecols = [norm_to_original[c] for c in wanted_cols if c in norm_to_original]
        if not usecols:
            usecols = None

    df = pd.read_csv(path, low_memory=False, usecols=usecols, nrows=nrows)
    return canonicalize_common_columns(df)


# -----------------------------
# Kidney feature engineering
# -----------------------------


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    if isinstance(b, pd.Series):
        b = b.replace({0: np.nan})
    elif b == 0:
        b = np.nan
    return a / b



def map_sex_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.lower()
    return s.map({"m": 1, "male": 1, "f": 0, "female": 0})



def estimate_egfr_2021(scr_mg_dl: pd.Series, age: pd.Series, sex_series: Optional[pd.Series]) -> pd.Series:
    """
    CKD-EPI 2021 race-free creatinine equation approximation.
    """
    scr = pd.to_numeric(scr_mg_dl, errors="coerce")
    age = pd.to_numeric(age, errors="coerce")
    if sex_series is None:
        female = pd.Series(np.zeros(len(scr)), index=scr.index)
    else:
        female = (map_sex_numeric(sex_series) == 0).astype(float)

    k = np.where(female == 1, 0.7, 0.9)
    alpha = np.where(female == 1, -0.241, -0.302)
    min_term = np.minimum(scr / k, 1.0)
    max_term = np.maximum(scr / k, 1.0)
    egfr = 142.0 * (min_term ** alpha) * (max_term ** -1.200) * (0.9938 ** age) * (1.012 ** female)
    return pd.Series(egfr, index=scr.index)



def compute_aki_stage(creatinine_baseline: pd.Series, creatinine_peak: pd.Series) -> pd.Series:
    base = pd.to_numeric(creatinine_baseline, errors="coerce")
    peak = pd.to_numeric(creatinine_peak, errors="coerce")
    delta = peak - base
    ratio = peak / base.replace({0: np.nan})

    stage = pd.Series(np.zeros(len(base), dtype=float), index=base.index)
    stage[(delta >= 0.3) | (ratio >= 1.5)] = 1
    stage[ratio >= 2.0] = 2
    stage[(ratio >= 3.0) | (peak >= 4.0)] = 3
    return stage.fillna(0).astype(int)



def engineer_kidney_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    age_col = find_col(df, ["age", "age_years"])
    sex_col = find_col(df, ["sex", "gender"])

    scr_base_col = find_col(
        df,
        [
            "serum_creatinine_baseline",
            "baseline_creatinine",
            "creatinine_baseline",
            "scr_baseline",
            "creatinine_base",
            "baseline_scr",
        ],
    )
    scr_peak_col = find_col(
        df,
        [
            "serum_creatinine_peak",
            "peak_creatinine",
            "creatinine_peak",
            "scr_peak",
            "max_creatinine",
            "creatinine",
        ],
    )
    bun_col = find_col(df, ["bun", "blood_urea_nitrogen"])
    potassium_col = find_col(df, ["potassium", "k", "serum_potassium"])
    phosphate_col = find_col(df, ["phosphate", "phosphorus"])
    magnesium_col = find_col(df, ["magnesium", "mg"])
    proteinuria_col = find_col(df, ["proteinuria", "urine_protein", "proteinuria_g_dl"])
    hematuria_col = find_col(df, ["hematuria", "blood_in_urine", "urine_blood"])
    urine_na_col = find_col(df, ["urine_sodium", "urine_na", "una"])
    urine_cr_col = find_col(df, ["urine_creatinine", "ucr"])
    serum_na_col = find_col(df, ["serum_sodium", "sodium", "na"])
    urine_osm_col = find_col(df, ["urine_osmolality", "uosm"])
    egfr_col = find_col(df, ["egfr", "e_gfr"])

    if scr_peak_col is not None and scr_base_col is None:
        scr_base_col = scr_peak_col

    if scr_base_col is not None:
        df["kidney_creatinine_baseline"] = pd.to_numeric(df[scr_base_col], errors="coerce")
    else:
        df["kidney_creatinine_baseline"] = np.nan

    if scr_peak_col is not None:
        df["kidney_creatinine_peak"] = pd.to_numeric(df[scr_peak_col], errors="coerce")
    else:
        df["kidney_creatinine_peak"] = df["kidney_creatinine_baseline"]

    df["kidney_creatinine_delta"] = df["kidney_creatinine_peak"] - df["kidney_creatinine_baseline"]
    df["kidney_creatinine_ratio"] = safe_div(df["kidney_creatinine_peak"], df["kidney_creatinine_baseline"])
    df["kidney_creatinine_rise_gt_1"] = (df["kidney_creatinine_delta"] > 1.0).astype(float)

    if age_col is not None:
        age_series = pd.to_numeric(df[age_col], errors="coerce")
    else:
        age_series = pd.Series(np.nan, index=df.index)

    if egfr_col is not None:
        df["kidney_egfr_baseline"] = pd.to_numeric(df[egfr_col], errors="coerce")
    else:
        df["kidney_egfr_baseline"] = estimate_egfr_2021(
            df["kidney_creatinine_baseline"],
            age_series,
            df[sex_col] if sex_col else None,
        )

    df["kidney_egfr_current"] = estimate_egfr_2021(
        df["kidney_creatinine_peak"],
        age_series,
        df[sex_col] if sex_col else None,
    )
    df["kidney_egfr_decline"] = df["kidney_egfr_baseline"] - df["kidney_egfr_current"]
    df["kidney_egfr_decline_pct"] = 100.0 * safe_div(df["kidney_egfr_decline"], df["kidney_egfr_baseline"])

    if bun_col is not None:
        df["kidney_bun"] = pd.to_numeric(df[bun_col], errors="coerce")
    else:
        df["kidney_bun"] = np.nan
    df["kidney_bun_cr_ratio"] = safe_div(df["kidney_bun"], df["kidney_creatinine_peak"])

    if potassium_col is not None:
        df["kidney_potassium"] = pd.to_numeric(df[potassium_col], errors="coerce")
    else:
        df["kidney_potassium"] = np.nan
    df["kidney_hyperkalemia"] = (df["kidney_potassium"] >= 5.5).astype(float)
    df["kidney_critical_hyperkalemia"] = (df["kidney_potassium"] >= 6.5).astype(float)

    df["kidney_phosphate"] = pd.to_numeric(df[phosphate_col], errors="coerce") if phosphate_col else np.nan
    df["kidney_magnesium"] = pd.to_numeric(df[magnesium_col], errors="coerce") if magnesium_col else np.nan
    df["kidney_proteinuria"] = pd.to_numeric(df[proteinuria_col], errors="coerce") if proteinuria_col else np.nan

    if hematuria_col:
        if pd.api.types.is_numeric_dtype(df[hematuria_col]):
            df["kidney_hematuria"] = pd.to_numeric(df[hematuria_col], errors="coerce").fillna(0)
        else:
            hs = df[hematuria_col].astype(str).str.lower().str.strip()
            df["kidney_hematuria"] = hs.isin(["1", "true", "yes", "positive", "present"]).astype(float)
    else:
        df["kidney_hematuria"] = np.nan

    if urine_na_col and urine_cr_col and serum_na_col:
        urine_na = pd.to_numeric(df[urine_na_col], errors="coerce")
        urine_cr = pd.to_numeric(df[urine_cr_col], errors="coerce")
        serum_na = pd.to_numeric(df[serum_na_col], errors="coerce")
        serum_cr = pd.to_numeric(df["kidney_creatinine_peak"], errors="coerce")
        df["kidney_fena_pct"] = 100.0 * safe_div(urine_na * serum_cr, serum_na * urine_cr)
    else:
        df["kidney_fena_pct"] = np.nan

    df["kidney_prerenal_pattern"] = (df["kidney_fena_pct"] < 1.0).astype(float)
    df["kidney_intrinsic_pattern"] = (df["kidney_fena_pct"] > 2.0).astype(float)
    df["kidney_aki_stage"] = compute_aki_stage(df["kidney_creatinine_baseline"], df["kidney_creatinine_peak"])
    df["kidney_aki_stage3"] = (df["kidney_aki_stage"] >= 3).astype(float)
    df["kidney_nephrology_alert"] = (
        (df["kidney_creatinine_peak"] > 4.0)
        | (df["kidney_critical_hyperkalemia"] > 0)
        | (df["kidney_creatinine_delta"] > 1.0)
    ).astype(float)

    if urine_osm_col:
        df["kidney_urine_osmolality"] = pd.to_numeric(df[urine_osm_col], errors="coerce")
    else:
        df["kidney_urine_osmolality"] = np.nan

    if age_col is not None:
        age_num = pd.to_numeric(df[age_col], errors="coerce")
        df["kidney_creatinine_x_age"] = df["kidney_creatinine_peak"] * age_num
    else:
        df["kidney_creatinine_x_age"] = np.nan

    wbc_col = find_col(df, ["wbc", "white_blood_cells", "white_cell_count"])
    if wbc_col is not None:
        wbc_num = pd.to_numeric(df[wbc_col], errors="coerce")
        df["kidney_proteinuria_x_wbc"] = df["kidney_proteinuria"] * wbc_num
        df["kidney_wbc_abnormal"] = ((wbc_num < 4.0) | (wbc_num > 12.0)).astype(float)
    else:
        df["kidney_proteinuria_x_wbc"] = np.nan
        df["kidney_wbc_abnormal"] = np.nan

    return df


# -----------------------------
# MIMIC raw processing
# -----------------------------


def assign_mimic_feature(label: str) -> Optional[str]:
    s = normalize_name(label)

    if "urine" in s and "creatinine" in s:
        return "urine_creatinine"
    if "urine" in s and "sodium" in s:
        return "urine_sodium"
    if "urine" in s and ("osmolality" in s or "osm" in s):
        return "urine_osmolality"
    if "proteinuria" in s or ("urine" in s and "protein" in s):
        return "proteinuria"
    if "hematuria" in s or ("urine" in s and "blood" in s):
        return "hematuria"

    if "creatinine" in s:
        return "creatinine"
    if "blood_urea_nitrogen" in s or "urea_nitrogen" in s or s == "bun" or "bun" in s:
        return "bun"
    if "potassium" in s:
        return "potassium"
    if "phosphate" in s or "phosphorus" in s:
        return "phosphate"
    if "magnesium" in s:
        return "magnesium"
    if "sodium" in s:
        return "sodium"
    if s == "wbc" or "white_blood" in s:
        return "wbc"
    if s == "heart_rate" or s == "hr":
        return "heart_rate"
    if ("systolic" in s and "blood_pressure" in s) or s == "sbp":
        return "sbp"
    if ("diastolic" in s and "blood_pressure" in s) or s == "dbp":
        return "dbp"
    if s in {"mean_bp", "map", "mean_arterial_pressure"} or ("mean" in s and "blood_pressure" in s):
        return "map"
    if s in {"respiratory_rate", "resp_rate", "rr"}:
        return "resp_rate"
    if s in {"temperature", "temp", "temperature_c", "temperature_f"}:
        return "temperature"
    if s in {"spo2", "o2_sat", "oxygen_saturation"} or ("oxygen" in s and "saturation" in s):
        return "spo2"
    if s in {"urine_output", "urine_out"} or ("urine" in s and "output" in s):
        return "urine_output"
    if s in {"glucose", "blood_glucose"}:
        return "glucose"
    return None



def parse_hematuria_value(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    positive_terms = {
        "positive",
        "present",
        "large",
        "moderate",
        "small",
        "trace",
        "true",
        "yes",
        "1",
    }
    negative_terms = {"negative", "none", "absent", "false", "no", "0"}
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[s.isin(positive_terms)] = 1.0
    out[s.isin(negative_terms)] = 0.0
    return out



def load_mimic_events(chartevents_path: str, d_items_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    ce = read_csv_selected(
        chartevents_path,
        wanted_cols=[
            "subject_id",
            "hadm_id",
            "icustay_id",
            "itemid",
            "charttime",
            "storetime",
            "valuenum",
            "value",
            "valueuom",
        ],
        nrows=max_rows,
    )
    di = read_csv_selected(
        d_items_path,
        wanted_cols=[
            "itemid",
            "label",
            "abbreviation",
            "category",
            "unitname",
            "param_type",
            "linksto",
            "dbsource",
        ],
    )

    ce_item = find_col(ce, ["ITEMID"])
    di_item = find_col(di, ["ITEMID"])
    if ce_item is None or di_item is None:
        raise ValueError(
            "Expected ITEMID in both CHARTEVENTS and D_ITEMS. "
            f"Got CHARTEVENTS columns={list(ce.columns)} ; D_ITEMS columns={list(di.columns)}"
        )

    label_col = find_col(di, ["LABEL", "ABBREVIATION"])
    if label_col is None:
        raise ValueError(f"D_ITEMS is missing LABEL/ABBREVIATION. Columns={list(di.columns)}")

    meta_cols = [c for c in [di_item, label_col, find_col(di, ["CATEGORY"]), find_col(di, ["UNITNAME"])] if c is not None]
    di_small = di[meta_cols].copy()
    di_small = di_small.rename(columns={di_item: "ITEMID", label_col: "LABEL"})

    ce2 = ce.rename(columns={ce_item: "ITEMID"}).copy()
    merged = ce2.merge(di_small, on="ITEMID", how="left")

    time_col = find_col(merged, ["CHARTTIME", "STORETIME"])
    if time_col is None:
        raise ValueError(f"CHARTEVENTS is missing CHARTTIME/STORETIME. Columns={list(merged.columns)}")

    merged["charttime"] = pd.to_datetime(merged[time_col], errors="coerce")
    merged = merged.dropna(subset=["charttime"]).copy()

    valuenum_col = find_col(merged, ["VALUENUM"])
    value_col = find_col(merged, ["VALUE"])
    merged["numeric_value"] = pd.to_numeric(merged[valuenum_col], errors="coerce") if valuenum_col else np.nan
    if value_col is not None:
        numeric_from_text = pd.to_numeric(merged[value_col], errors="coerce")
        merged["numeric_value"] = merged["numeric_value"].fillna(numeric_from_text)

    merged["feature_name"] = merged["LABEL"].fillna("").map(assign_mimic_feature)

    hem_mask = merged["feature_name"] == "hematuria"
    if value_col is not None and hem_mask.any():
        merged.loc[hem_mask, "numeric_value"] = merged.loc[hem_mask, "numeric_value"].fillna(
            parse_hematuria_value(merged.loc[hem_mask, value_col])
        )

    key_parts = []
    for col_name in ["ICUSTAY_ID", "HADM_ID", "SUBJECT_ID"]:
        c = find_col(merged, [col_name])
        if c is not None:
            key_parts.append(c)

    if not key_parts:
        raise ValueError(
            "Could not find any patient/stay identifier in CHARTEVENTS. "
            f"Columns={list(merged.columns)}"
        )

    def make_encounter_key(frame: pd.DataFrame) -> pd.Series:
        parts = []
        for c in key_parts:
            parts.append(frame[c].astype(str).fillna("nan"))
        out = parts[0].copy()
        for p in parts[1:]:
            out = out + "|" + p
        return out

    merged["encounter_key"] = make_encounter_key(merged)

    keep_cols = [
        c
        for c in [
            find_col(merged, ["SUBJECT_ID"]),
            find_col(merged, ["HADM_ID"]),
            find_col(merged, ["ICUSTAY_ID"]),
            "encounter_key",
            "charttime",
            "ITEMID",
            "LABEL",
            "feature_name",
            "numeric_value",
        ]
        if c is not None
    ]
    events = merged[keep_cols].copy()

    if events["feature_name"].notna().sum() == 0:
        top_labels = (
            merged["LABEL"].fillna("<missing>").astype(str).value_counts().head(20).index.tolist()
        )
        raise ValueError(
            "Could not map any D_ITEMS labels to features. "
            f"Example labels={top_labels}"
        )

    return events



def build_mimic_timeline_dataset(events: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in [find_col(events, ["SUBJECT_ID"]), find_col(events, ["HADM_ID"]), find_col(events, ["ICUSTAY_ID"])] if c]
    group_cols = ["encounter_key", "charttime"] + id_cols

    usable = events[events["feature_name"].notna()].copy()
    usable = usable.dropna(subset=["numeric_value"]).copy()

    if usable.empty:
        raise ValueError("No numeric feature rows remain after filtering raw MIMIC events.")

    timeline = (
        usable.groupby(group_cols + ["feature_name"], dropna=False)["numeric_value"]
        .mean()
        .unstack("feature_name")
        .reset_index()
    )

    timeline = timeline.sort_values(["encounter_key", "charttime"]).reset_index(drop=True)

    if "creatinine" in timeline.columns:
        timeline["creatinine_baseline"] = timeline.groupby("encounter_key")["creatinine"].transform("min")
        timeline["creatinine_peak"] = timeline["creatinine"]
        timeline["serum_creatinine_baseline"] = timeline["creatinine_baseline"]
        timeline["serum_creatinine_peak"] = timeline["creatinine_peak"]
    if "sodium" in timeline.columns:
        timeline["serum_sodium"] = timeline["sodium"]

    timeline = engineer_kidney_features(timeline)

    max_stage = timeline.groupby("encounter_key")["kidney_aki_stage"].transform("max")
    timeline["encounter_max_aki_stage"] = max_stage.astype(int)
    timeline["aki_stage_label"] = "stage_" + max_stage.astype(int).astype(str)
    timeline["severe_aki_label"] = np.where(max_stage >= 2, "severe", "not_severe")

    return timeline



def aggregate_mimic_encounter_dataset(timeline: pd.DataFrame) -> pd.DataFrame:
    timeline = timeline.sort_values(["encounter_key", "charttime"]).reset_index(drop=True)

    id_cols = [c for c in ["encounter_key", find_col(timeline, ["SUBJECT_ID"]), find_col(timeline, ["HADM_ID"]), find_col(timeline, ["ICUSTAY_ID"])] if c]
    target_cols = [c for c in ["aki_stage_label", "severe_aki_label"] if c in timeline.columns]
    numeric_cols = [
        c
        for c in timeline.columns
        if pd.api.types.is_numeric_dtype(timeline[c])
        and c not in id_cols
        and c not in target_cols
    ]

    grouped = timeline.groupby("encounter_key", dropna=False)
    first_block = grouped[id_cols].first() if id_cols else pd.DataFrame(index=grouped.size().index)
    last_block = grouped[numeric_cols].last().add_suffix("_last") if numeric_cols else pd.DataFrame(index=grouped.size().index)
    mean_block = grouped[numeric_cols].mean().add_suffix("_mean") if numeric_cols else pd.DataFrame(index=grouped.size().index)
    max_block = grouped[numeric_cols].max().add_suffix("_max") if numeric_cols else pd.DataFrame(index=grouped.size().index)
    min_block = grouped[numeric_cols].min().add_suffix("_min") if numeric_cols else pd.DataFrame(index=grouped.size().index)
    n_timepoints = grouped.size().rename("n_timepoints").to_frame()

    parts = [first_block, n_timepoints, last_block, mean_block, max_block, min_block]
    encounter = pd.concat(parts, axis=1).reset_index(drop=True)

    # Merge targets separately to avoid suffix collisions.
    target_df = grouped[target_cols].last().reset_index() if target_cols else pd.DataFrame({"encounter_key": grouped.size().index})
    if "encounter_key" not in encounter.columns:
        encounter = encounter.merge(target_df, on="encounter_key", how="left")
    else:
        encounter = encounter.merge(target_df, on="encounter_key", how="left")

    return encounter



def build_mimic_datasets(chartevents_path: str, d_items_path: str, max_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events = load_mimic_events(chartevents_path, d_items_path, max_rows=max_rows)
    timeline = build_mimic_timeline_dataset(events)
    encounter = aggregate_mimic_encounter_dataset(timeline)
    return timeline, encounter


# -----------------------------
# Data preparation
# -----------------------------


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, [
        "pathogen",
        "organism",
        "pathogen_label",
        "organism_label",
        "label",
        "target",
        "y",
        "aki_stage_label",
        "severe_aki_label",
    ])



def infer_patient_id_column(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, [
        "encounter_key",
        "patient_id",
        "subject_id",
        "hadm_id",
        "icustay_id",
        "stay_id",
        "encounter_id",
        "admission_id",
    ])



def infer_time_column(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["charttime", "event_time", "timestamp", "time", "datetime", "specimen_time"])



def exclude_columns(df: pd.DataFrame, target_col: str, patient_id_col: Optional[str], time_col: Optional[str]) -> List[str]:
    exclude = {target_col}
    if patient_id_col:
        exclude.add(patient_id_col)
    if time_col:
        exclude.add(time_col)
    return [c for c in df.columns if c not in exclude]



def one_hot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=cat_cols, dummy_na=True)



def choose_stratify(y: np.ndarray) -> Optional[np.ndarray]:
    vals, counts = np.unique(y, return_counts=True)
    if len(vals) <= 1:
        return None
    if counts.min() < 2:
        return None
    return y


@dataclass
class TabularPack:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    label_encoder: LabelEncoder
    scaler: StandardScaler
    imputer: SimpleImputer


@dataclass
class SequencePack:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    lengths_train: np.ndarray
    lengths_val: np.ndarray
    lengths_test: np.ndarray
    feature_names: List[str]
    label_encoder: LabelEncoder
    scaler: StandardScaler
    imputer: SimpleImputer


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, lengths: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.lengths[idx]



def prepare_tabular_data(
    df: pd.DataFrame,
    target_col: str,
    patient_id_col: Optional[str],
    time_col: Optional[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> TabularPack:
    use_cols = exclude_columns(df, target_col, patient_id_col, time_col)
    X_df = one_hot_dataframe(df[use_cols])
    y_raw = df[target_col].astype(str).fillna("unknown")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    strat_all = choose_stratify(y)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=strat_all,
    )

    rel_val = val_size / (1.0 - test_size)
    strat_train = choose_stratify(y_train)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_df,
        y_train,
        test_size=rel_val,
        random_state=seed,
        stratify=strat_train,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train_df)
    X_val = imputer.transform(X_val_df)
    X_test = imputer.transform(X_test_df)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return TabularPack(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        y_val=np.asarray(y_val, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_names=list(X_df.columns),
        label_encoder=le,
        scaler=scaler,
        imputer=imputer,
    )



def build_sequences(
    df: pd.DataFrame,
    target_col: str,
    patient_id_col: str,
    time_col: str,
    max_len: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feature_cols = exclude_columns(df, target_col, patient_id_col, time_col)
    X_df = one_hot_dataframe(df[feature_cols])
    all_df = pd.concat([df[[patient_id_col, time_col, target_col]].copy(), X_df], axis=1)
    all_df[time_col] = pd.to_datetime(all_df[time_col], errors="coerce")
    all_df = all_df.dropna(subset=[time_col]).sort_values([patient_id_col, time_col], ascending=[True, True])

    patient_groups = []
    targets = []
    lengths = []
    feature_names = list(X_df.columns)

    for pid, g in all_df.groupby(patient_id_col):
        g = g.tail(max_len)
        seq = g[feature_names].to_numpy(dtype=np.float32)
        target = str(g[target_col].iloc[-1])
        lengths.append(len(seq))
        patient_groups.append(seq)
        targets.append(target)

    if not patient_groups:
        raise ValueError("No sequences were built. Check patient/time columns and missing timestamps.")

    max_actual_len = max(lengths)
    feature_dim = len(feature_names)
    X = np.zeros((len(patient_groups), max_actual_len, feature_dim), dtype=np.float32)
    for i, seq in enumerate(patient_groups):
        X[i, : len(seq), :] = seq

    return X, np.array(targets), np.array(lengths, dtype=np.int64), feature_names



def prepare_sequence_data(
    df: pd.DataFrame,
    target_col: str,
    patient_id_col: str,
    time_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    max_len: int = 16,
) -> SequencePack:
    X, y_raw, lengths, feature_names = build_sequences(df, target_col, patient_id_col, time_col, max_len=max_len)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    idx = np.arange(len(y))
    strat_all = choose_stratify(y)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat_all)
    rel_val = val_size / (1.0 - test_size)
    strat_train = choose_stratify(y[train_idx])
    train_idx, val_idx = train_test_split(train_idx, test_size=rel_val, random_state=seed, stratify=strat_train)

    X_train_flat = X[train_idx].reshape(-1, X.shape[-1])
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_flat = imputer.fit_transform(X_train_flat)
    X_train_flat = scaler.fit_transform(X_train_flat)

    def transform_block(block: np.ndarray) -> np.ndarray:
        flat = block.reshape(-1, block.shape[-1])
        flat = imputer.transform(flat)
        flat = scaler.transform(flat)
        return flat.reshape(block.shape[0], block.shape[1], block.shape[2]).astype(np.float32)

    return SequencePack(
        X_train=transform_block(X[train_idx]),
        X_val=transform_block(X[val_idx]),
        X_test=transform_block(X[test_idx]),
        y_train=y[train_idx].astype(np.int64),
        y_val=y[val_idx].astype(np.int64),
        y_test=y[test_idx].astype(np.int64),
        lengths_train=lengths[train_idx],
        lengths_val=lengths[val_idx],
        lengths_test=lengths[test_idx],
        feature_names=feature_names,
        label_encoder=le,
        scaler=scaler,
        imputer=imputer,
    )


# -----------------------------
# Models
# -----------------------------


class RationalActivation(nn.Module):
    """
    n(a, b, lam, z) = lam * (z + a z^3) / (1 + b z^2)
    Default Padé params: a=1/15, b=2/5, lam=1.
    """

    def __init__(self, a: float = 1.0 / 15.0, b: float = 2.0 / 5.0, lam: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.a = nn.Parameter(torch.tensor(float(a), dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(float(b), dtype=torch.float32))
            self.lam = nn.Parameter(torch.tensor(float(lam), dtype=torch.float32))
        else:
            self.register_buffer("a", torch.tensor(float(a), dtype=torch.float32))
            self.register_buffer("b", torch.tensor(float(b), dtype=torch.float32))
            self.register_buffer("lam", torch.tensor(float(lam), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x * x
        x3 = x2 * x
        return self.lam * (x + self.a * x3) / (1.0 + self.b * x2 + 1e-8)


class DNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Sequence[int], dropout: float = 0.20):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(RationalActivation())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.10):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            RationalActivation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        last = h_n[-1]
        return self.head(last)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            RationalActivation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]
        return self.head(last)


# -----------------------------
# Training and calibration
# -----------------------------


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    weights = np.zeros(classes.max() + 1, dtype=np.float32)
    total = counts.sum()
    for c, n in zip(classes, counts):
        weights[c] = total / (len(classes) * max(n, 1))
    return torch.tensor(weights, dtype=torch.float32)



def memory_aware_lr(epoch: int, total_epochs: int, peak_lr: float = 0.06, warmup_ratio: float = 0.10, decay_ratio: float = 0.80, alpha: float = 0.1) -> float:
    t1 = max(1, int(total_epochs * warmup_ratio))
    t2 = max(t1 + 1, int(total_epochs * decay_ratio))
    t = epoch + 1
    if t <= t1:
        return peak_lr * (t / t1)
    if t <= t2:
        return peak_lr * ((t2 - t) / max(1, (t2 - t1)))
    return peak_lr * alpha * ((t2 - t1) / max(1, (t - t2 + 1)))


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> "TemperatureScaler":
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self



def evaluate_logits(logits: np.ndarray, y_true: np.ndarray, label_encoder: LabelEncoder) -> Dict[str, object]:
    y_pred = logits.argmax(axis=1)
    labels = np.arange(len(label_encoder.classes_))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "pred_indices": y_pred.tolist(),
        "pred_labels": label_encoder.inverse_transform(y_pred).tolist(),
    }



def train_tabular_model(
    pack: TabularPack,
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    weight_decay: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object], np.ndarray]:
    num_classes = len(pack.label_encoder.classes_)
    model = DNNClassifier(input_dim=pack.X_train.shape[1], num_classes=num_classes, hidden_dims=hidden_dims).to(device)

    class_weights = compute_class_weights(pack.y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.06, weight_decay=weight_decay)

    train_loader = DataLoader(TabularDataset(pack.X_train, pack.y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(pack.X_val, pack.y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TabularDataset(pack.X_test, pack.y_test), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = -1.0
    history = []

    for epoch in range(epochs):
        lr = memory_aware_lr(epoch, epochs, peak_lr=0.06)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_logits_all = []
        val_y_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                val_logits_all.append(logits)
                val_y_all.append(yb.numpy())
        val_logits_all = np.concatenate(val_logits_all)
        val_y_all = np.concatenate(val_y_all)
        val_metrics = evaluate_logits(val_logits_all, val_y_all, pack.label_encoder)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_macro_f1": val_metrics["macro_f1"], "lr": lr})

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits_tensor = torch.tensor(model(torch.tensor(pack.X_val, dtype=torch.float32).to(device)).cpu().numpy(), dtype=torch.float32)
    val_labels_tensor = torch.tensor(pack.y_val, dtype=torch.long)
    scaler = TemperatureScaler().fit(val_logits_tensor, val_labels_tensor)

    test_logits = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb).cpu()
            logits = scaler(logits)
            test_logits.append(logits.numpy())
    test_logits = np.concatenate(test_logits)

    metrics = evaluate_logits(test_logits, pack.y_test, pack.label_encoder)
    metrics["history"] = history
    metrics["temperature"] = float(scaler.temperature.detach().cpu().item())
    return model, metrics, test_logits



def train_sequence_model(
    pack: SequencePack,
    model_name: str,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object], np.ndarray]:
    num_classes = len(pack.label_encoder.classes_)
    input_dim = pack.X_train.shape[-1]

    if model_name == "rnn":
        model = RNNClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    elif model_name == "lstm":
        model = LSTMClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    else:
        raise ValueError(f"Unsupported sequence model: {model_name}")

    class_weights = compute_class_weights(pack.y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)

    train_loader = DataLoader(SequenceDataset(pack.X_train, pack.y_train, pack.lengths_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(pack.X_val, pack.y_val, pack.lengths_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(pack.X_test, pack.y_test, pack.lengths_test), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val = -1.0
    history = []

    for epoch in range(epochs):
        lr = memory_aware_lr(epoch, epochs, peak_lr=0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        train_losses = []
        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_logits_all = []
        val_y_all = []
        with torch.no_grad():
            for xb, yb, lb in val_loader:
                xb, lb = xb.to(device), lb.to(device)
                logits = model(xb, lb).cpu().numpy()
                val_logits_all.append(logits)
                val_y_all.append(yb.numpy())
        val_logits_all = np.concatenate(val_logits_all)
        val_y_all = np.concatenate(val_y_all)
        val_metrics = evaluate_logits(val_logits_all, val_y_all, pack.label_encoder)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_macro_f1": val_metrics["macro_f1"], "lr": lr})

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    val_logits_chunks = []
    with torch.no_grad():
        for xb, _, lb in val_loader:
            xb, lb = xb.to(device), lb.to(device)
            val_logits_chunks.append(model(xb, lb).cpu())
    val_logits_tensor = torch.cat(val_logits_chunks, dim=0)
    val_labels_tensor = torch.tensor(pack.y_val, dtype=torch.long)
    scaler = TemperatureScaler().fit(val_logits_tensor, val_labels_tensor)

    test_logits = []
    with torch.no_grad():
        for xb, _, lb in test_loader:
            xb, lb = xb.to(device), lb.to(device)
            logits = scaler(model(xb, lb).cpu())
            test_logits.append(logits.numpy())
    test_logits = np.concatenate(test_logits)

    metrics = evaluate_logits(test_logits, pack.y_test, pack.label_encoder)
    metrics["history"] = history
    metrics["temperature"] = float(scaler.temperature.detach().cpu().item())
    return model, metrics, test_logits


# -----------------------------
# Export helpers
# -----------------------------


def save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)



def save_predictions(path: str, y_true: np.ndarray, logits: np.ndarray, le: LabelEncoder) -> None:
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    out = pd.DataFrame(
        {
            "y_true_idx": y_true,
            "y_true_label": le.inverse_transform(y_true),
            "y_pred_idx": pred_idx,
            "y_pred_label": le.inverse_transform(pred_idx),
            "y_pred_confidence": probs.max(axis=1),
        }
    )
    for i, cls in enumerate(le.classes_):
        out[f"prob_{cls}"] = probs[:, i]
    out.to_csv(path, index=False)



def persist_artifacts(
    output_dir: str,
    prefix: str,
    metrics: Dict[str, object],
    test_logits: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    model: nn.Module,
    feature_names: List[str],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_json(os.path.join(output_dir, f"{prefix}_metrics.json"), metrics)
    save_predictions(os.path.join(output_dir, f"{prefix}_predictions.csv"), y_test, test_logits, label_encoder)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(label_encoder.classes_),
            "feature_names": feature_names,
        },
        os.path.join(output_dir, f"{prefix}_model.pt"),
    )


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified PenuX kidney-aware DNN/RNN/LSTM pipeline")

    parser.add_argument("--csv", default=None, help="Prepared tabular CSV input")
    parser.add_argument("--chartevents", default=None, help="MIMIC-III CHARTEVENTS.csv path")
    parser.add_argument("--d-items", default=None, help="MIMIC-III D_ITEMS.csv path")
    parser.add_argument("--mimic-max-rows", type=int, default=None, help="Optional row cap when reading CHARTEVENTS")

    parser.add_argument("--target", default=None, help="Target column. For raw MIMIC, common choices: aki_stage_label or severe_aki_label")
    parser.add_argument("--model", choices=["dnn", "rnn", "lstm"], default="dnn", help="Neural model family")
    parser.add_argument("--patient-id", default=None, help="Patient identifier column for sequence models")
    parser.add_argument("--time-col", default=None, help="Time column for sequence models")
    parser.add_argument("--max-seq-len", type=int, default=16, help="Maximum sequence length per patient")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dims", default="128,64", help="Comma-separated hidden dims for DNN")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden size for RNN/LSTM")
    parser.add_argument("--num-layers", type=int, default=1, help="Recurrent depth for RNN/LSTM")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def load_input_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, object]]:
    """
    Returns:
        working_df: the dataframe to train on for the chosen model
        aux_df: optional second dataframe (timeline or encounter) for export/reference
        source_meta: summary metadata
    """
    if args.csv:
        df = pd.read_csv(args.csv, low_memory=False)
        df = canonicalize_common_columns(df)
        df = engineer_kidney_features(df)
        return df, None, {"input_mode": "prepared_csv", "source_csv": os.path.abspath(args.csv)}

    if args.chartevents and args.d_items:
        timeline_df, encounter_df = build_mimic_datasets(args.chartevents, args.d_items, max_rows=args.mimic_max_rows)
        if args.model == "dnn":
            working_df = encounter_df
            aux_df = timeline_df
        else:
            working_df = timeline_df
            aux_df = encounter_df
        return working_df, aux_df, {
            "input_mode": "mimic_raw",
            "chartevents": os.path.abspath(args.chartevents),
            "d_items": os.path.abspath(args.d_items),
            "mimic_max_rows": args.mimic_max_rows,
        }

    raise ValueError("Provide either --csv OR both --chartevents and --d-items.")



def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df, aux_df, source_meta = load_input_data(args)

    if source_meta["input_mode"] == "mimic_raw":
        if args.model == "dnn":
            df.to_csv(os.path.join(args.output_dir, "mimic_prepared_dataset.csv"), index=False)
            if aux_df is not None:
                aux_df.to_csv(os.path.join(args.output_dir, "mimic_timeline_dataset.csv"), index=False)
        else:
            df.to_csv(os.path.join(args.output_dir, "mimic_timeline_dataset.csv"), index=False)
            if aux_df is not None:
                aux_df.to_csv(os.path.join(args.output_dir, "mimic_prepared_dataset.csv"), index=False)

    target_col = args.target or infer_target_column(df)
    if target_col is None:
        raise ValueError(
            "Could not infer target column. Pass --target explicitly. "
            "For raw MIMIC, use --target aki_stage_label or --target severe_aki_label."
        )

    patient_id_col = args.patient_id or infer_patient_id_column(df)
    time_col = args.time_col or infer_time_column(df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "dnn":
        pack = prepare_tabular_data(df, target_col, patient_id_col, time_col, seed=args.seed)
        hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]
        model, metrics, test_logits = train_tabular_model(
            pack=pack,
            hidden_dims=hidden_dims,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            device=device,
        )
        prefix = f"penux_kidney_{args.model}"
        persist_artifacts(args.output_dir, prefix, metrics, test_logits, pack.y_test, pack.label_encoder, model, pack.feature_names)
        save_json(
            os.path.join(args.output_dir, f"{prefix}_config.json"),
            {
                "model": args.model,
                "target_col": target_col,
                "input_dim": len(pack.feature_names),
                "num_classes": len(pack.label_encoder.classes_),
                "hidden_dims": hidden_dims,
                "device": str(device),
                **source_meta,
            },
        )
    else:
        if patient_id_col is None or time_col is None:
            raise ValueError(
                f"Model '{args.model}' requires a patient identifier and time column. "
                "Pass --patient-id and --time-col if they are not inferred automatically."
            )
        pack = prepare_sequence_data(
            df,
            target_col=target_col,
            patient_id_col=patient_id_col,
            time_col=time_col,
            seed=args.seed,
            max_len=args.max_seq_len,
        )
        model, metrics, test_logits = train_sequence_model(
            pack=pack,
            model_name=args.model,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            device=device,
        )
        prefix = f"penux_kidney_{args.model}"
        persist_artifacts(args.output_dir, prefix, metrics, test_logits, pack.y_test, pack.label_encoder, model, pack.feature_names)
        save_json(
            os.path.join(args.output_dir, f"{prefix}_config.json"),
            {
                "model": args.model,
                "target_col": target_col,
                "patient_id_col": patient_id_col,
                "time_col": time_col,
                "input_dim": len(pack.feature_names),
                "num_classes": len(pack.label_encoder.classes_),
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "max_seq_len": args.max_seq_len,
                "device": str(device),
                **source_meta,
            },
        )

    summary = {
        "status": "ok",
        "output_dir": os.path.abspath(args.output_dir),
        "model": args.model,
        "target_col": target_col,
        **source_meta,
    }
    save_json(os.path.join(args.output_dir, "run_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
