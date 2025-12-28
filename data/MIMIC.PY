#!/usr/bin/env python3
# mimic3_resistance_pipeline.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# FIXED CLASS ORDER
# ===============================
CLASSES = [
    "B:PSEUDOMONAS AERUGINOSA",
    "B:STAPH AUREUS COAG +",
    "B:SERRATIA MARCESCENS",
    "B:MORGANELLA MORGANII",
    "B:ESCHERICHIA COLI",
    "B:PROTEUS MIRABILIS",
    "B:PROVIDENCIA STUARTII",
    "B:POSITIVE FOR METHICILLIN RESISTANT STAPH AUREUS",
    "B:YEAST",
    "B:GRAM POSITIVE COCCUS(COCCI)",
    "B:OTHER",
    "V:OTHER"
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}

# ===============================
# LOAD MICROBIOLOGYEVENTS (columns lowercase)
# ===============================
MICRO_PATH = "dataset/mimic/mimic-iii-clinical-database-demo-1.4/MICROBIOLOGYEVENTS.csv"
micro_df = pd.read_csv(MICRO_PATH, low_memory=False)
micro_df.columns = micro_df.columns.str.lower()
micro_df = micro_df[["hadm_id", "spec_type_desc", "org_name", "interpretation"]].dropna(subset=["org_name"])

# MAP ORGANISM TO CLASS
def map_org(org):
    o = org.upper()
    if "PSEUDOMONAS AERUGINOSA" in o: return CLASSES[0]
    if "STAPH AUREUS" in o and "METHICILLIN" not in o: return CLASSES[1]
    if "SERRATIA MARCESCENS" in o: return CLASSES[2]
    if "MORGANELLA MORGANII" in o: return CLASSES[3]
    if "ESCHERICHIA COLI" in o: return CLASSES[4]
    if "PROTEUS MIRABILIS" in o: return CLASSES[5]
    if "PROVIDENCIA STUARTII" in o: return CLASSES[6]
    if "MRSA" in o or "METHICILLIN" in o: return CLASSES[7]
    if "YEAST" in o or "CANDIDA" in o: return CLASSES[8]
    if "COCCUS" in o or "COCCI" in o: return CLASSES[9]
    return CLASSES[10]

micro_df["label"] = micro_df["org_name"].apply(map_org)
micro_df = micro_df[micro_df["label"].isin(CLASSES)]

# ===============================
# LOAD PRESCRIPTIONS (columns lowercase)
# ===============================
PRESC_PATH = "dataset/mimic/mimic-iii-clinical-database-demo-1.4/PRESCRIPTIONS.csv"
presc_df = pd.read_csv(PRESC_PATH, low_memory=False)
presc_df.columns = presc_df.columns.str.lower()
antibiotics = ["VANCOMYCIN","CIPROFLOXACIN","MEROPENEM","PIPERACILLIN","CEFTRIAXONE"]
presc_df = presc_df[presc_df["drug"].str.upper().isin(antibiotics)]
presc_df = presc_df[["hadm_id", "drug"]]

# ONE-HOT PRESCRIPTIONS PER HADM
presc_onehot = pd.get_dummies(presc_df.set_index("hadm_id")["drug"]).groupby(level=0).max()
presc_onehot.reset_index(inplace=True)

# ===============================
# MERGE MICROBIOLOGY + PRESCRIPTIONS
# ===============================
merged_df = micro_df.merge(presc_onehot, on="hadm_id", how="left")
merged_df.fillna(0, inplace=True)

# FEATURES + TARGET
cat_cols = ["spec_type_desc", "interpretation"]
num_cols = [c for c in merged_df.columns if c not in cat_cols + ["hadm_id", "label"]]

# Encode categorical only
encoder = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])
X_encoded_cat = encoder.fit_transform(merged_df[cat_cols])

# Combine categorical + numeric prescription columns
X_encoded = np.hstack([X_encoded_cat, merged_df[num_cols].values])

# Target
y_onehot = tf.keras.utils.to_categorical(merged_df["label"].map(CLASS_TO_INDEX), num_classes=len(CLASSES))
hadm_ids = merged_df["hadm_id"].values

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test, hadm_train, hadm_test = train_test_split(
    X_encoded, y_onehot, hadm_ids, test_size=0.2, random_state=42
)

# ===============================
# DNN MODEL
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_encoded.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(CLASSES), activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# TRAIN
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=2)

# ===============================
# GENERAL ACCURACY
# ===============================
y_pred_labels = np.argmax(model.predict(X_test), axis=1)
y_true_labels = np.argmax(y_test, axis=1)
general_acc = (y_pred_labels == y_true_labels).mean()
print(f"\n=== GENERAL ACCURACY ON TEST SET: {general_acc:.4f} ===")

# ===============================
# PREDICTIONS PER HADM_ID
# ===============================
print("\n=== PREDICTIONS PER HADM_ID ===")
unique_hadm_ids = np.unique(hadm_test)
for hadm_id in unique_hadm_ids:
    idxs = np.where(hadm_test == hadm_id)[0]
    X_sub = X_test[idxs]
    preds = model.predict(X_sub)
    mean_probs = preds.mean(axis=0)
    print(f"\nHADM_ID: {hadm_id}")
    for cls, p in zip(CLASSES, mean_probs):
        print(f"{cls:65s} -> {p:.4f}")
