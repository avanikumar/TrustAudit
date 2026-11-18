"""LightGBM baseline model."""
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import load_dataset


def train_model(X_train, y_train, seed=42):
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, proba), accuracy_score(y, proba > 0.5)


if __name__ == "__main__":
    import sys

    ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "telco")
    model = train_model(ds.X_train, ds.y_train)
    for split, X, y in [("cal", ds.X_cal, ds.y_cal), ("test", ds.X_test, ds.y_test)]:
        auc, acc = evaluate(model, X, y)
        print(f"{ds.name} {split:4s} AUC={auc:.3f} acc={acc:.3f}")
