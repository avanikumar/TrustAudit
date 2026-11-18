"""
SHAP attributions for a single prediction.

Values are in the model's raw (log-odds) space: positive pushes the
prediction towards the positive class, negative pushes it away.
"""
import warnings

import numpy as np
import shap

# SHAP warns on every call that LightGBM output "changed to a list"; handled below.
warnings.filterwarnings("ignore", message="LightGBM binary classifier")


def build_explainer(model):
    return shap.TreeExplainer(model)


def attributions(explainer, X_row):
    """All features for one row, sorted by |SHAP|, with rank (1 = biggest) and share of total."""
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):          # some SHAP versions return one array per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (rows, features, classes) in newer versions
        sv = sv[:, :, 1]
    sv = sv[0]

    total = float(np.abs(sv).sum()) or 1.0
    order = np.argsort(-np.abs(sv))
    out = []
    for rank, i in enumerate(order, start=1):
        out.append({
            "feature": X_row.columns[i],
            "value": _plain(X_row.iloc[0, i]),
            "shap": float(sv[i]),
            "share": float(abs(sv[i]) / total),
            "rank": rank,
            "direction": "increases" if sv[i] > 0 else "decreases",
        })
    return out


def top_features(explainer, X_row, k=5):
    return attributions(explainer, X_row)[:k]


def _plain(v):
    """Make numpy / pandas scalars JSON-friendly."""
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v if isinstance(v, (int, str)) else str(v)
