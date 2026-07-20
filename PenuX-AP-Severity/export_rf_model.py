"""Export final Random Forest model trained on all 722 patients to JSON for JS inference."""
import json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("PenuX-AP-Severity/data/public_sanitized/ap_lnn_sanitized.csv")
LABEL = "严重程度"
X = df.drop(columns=[LABEL]).values.astype(np.float32)
y = df[LABEL].values.astype(np.float32)
FEATURE_NAMES = list(df.drop(columns=[LABEL]).columns)

# Train final model on all data (same hyperparams as train_ml_models.py)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)
rf.fit(X, y)

# Export each tree
def export_tree(tree, feature_names):
    t = tree.tree_
    def recurse(node):
        if t.feature[node] == -2:  # leaf
            v = t.value[node][0]
            return {"leaf": float(v[1] / v.sum())}
        return {
            "feature": int(t.feature[node]),
            "threshold": float(t.threshold[node]),
            "left": recurse(t.children_left[node]),
            "right": recurse(t.children_right[node])
        }
    return recurse(0)

trees = [export_tree(est, FEATURE_NAMES) for est in rf.estimators_]

out = {
    "n_features": len(FEATURE_NAMES),
    "feature_names": FEATURE_NAMES,
    "n_trees": len(trees),
    "threshold": 0.535,
    "trees": trees
}

with open("docs/penux_rf_model.json", "w") as f:
    json.dump(out, f, separators=(',', ':'))

import os
size_kb = os.path.getsize("docs/penux_rf_model.json") / 1024
print(f"Exported {len(trees)} trees · {len(FEATURE_NAMES)} features · {size_kb:.0f} KB")
