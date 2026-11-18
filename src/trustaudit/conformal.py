"""
Split conformal prediction for binary classification.

Nonconformity score = 1 - probability the model gives the true class.
Calibrating on held-out data gives prediction sets that contain the true
label with probability >= 1 - alpha (marginally, assuming exchangeability).
"""
import numpy as np

from .data import load_dataset
from .model import train_model


def calibrate(model, X_cal, y_cal, alpha=0.1):
    proba = model.predict_proba(X_cal)                       # shape (n, 2)
    scores = 1 - proba[np.arange(len(y_cal)), np.asarray(y_cal)]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)     # finite-sample correction
    return float(np.quantile(scores, level, method="higher"))


def predict_sets(model, X, q):
    proba = model.predict_proba(X)
    return proba >= 1 - q        # boolean (n, 2): is each class in the set?


def evaluate_sets(sets, y):
    y = np.asarray(y)
    covered = sets[np.arange(len(y)), y]      # True if the true class is in the set
    size = sets.sum(axis=1)                   # 0, 1 or 2 classes per set
    return covered.mean(), size.mean(), (size == 2).mean()


def describe_set(row_set, positive_label):
    """Turn a boolean [neg, pos] set into words for the explanation."""
    neg, pos = bool(row_set[0]), bool(row_set[1])
    if pos and not neg:
        return f"confident: {positive_label}"
    if neg and not pos:
        return f"confident: no {positive_label}"
    if pos and neg:
        return "uncertain: both outcomes are plausible"
    return "abstain: neither outcome reaches the confidence threshold"


if __name__ == "__main__":
    import sys

    ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "telco")
    model = train_model(ds.X_train, ds.y_train)
    for alpha in [0.05, 0.1, 0.2]:
        q = calibrate(model, ds.X_cal, ds.y_cal, alpha)
        cov, size, both = evaluate_sets(predict_sets(model, ds.X_test, q), ds.y_test)
        print(f"{ds.name} alpha={alpha:<5} target={1-alpha:.2f} coverage={cov:.3f} "
              f"avg_size={size:.2f} both={both:.2f}")
