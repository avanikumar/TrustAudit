"""
Dataset loading.

Every dataset is returned as a `Dataset` with a stratified 60/20/20
train / calibration / test split. The calibration split is held out from
training so conformal prediction gets an honest coverage guarantee.
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    name: str
    positive_label: str          # what y == 1 means, used in explanations
    X_train: pd.DataFrame
    X_cal: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_cal: pd.Series
    y_test: pd.Series


def _split(X, y, seed):
    # Two-step split: 60% train, then the remaining 40% into 20% cal / 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def load_telco(path="data/telco.csv", seed=42) -> Dataset:
    """IBM Telco customer churn (Kaggle: blastchar/telco-customer-churn)."""
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    # TotalCharges is stored as text and is blank for brand-new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return Dataset("telco", "churn", *_split(X, y, seed))


def load_cancer(seed=42) -> Dataset:
    """Wisconsin breast cancer (bundled with scikit-learn, no download needed).

    Relabelled so that y == 1 means malignant, which reads more naturally
    as the "risk" the model predicts.
    """
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data
    y = (bunch.target == 0).astype(int).rename("malignant")
    return Dataset("cancer", "malignancy", *_split(X, y, seed))


LOADERS = {"telco": load_telco, "cancer": load_cancer}


def load_dataset(name: str, **kwargs) -> Dataset:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)


if __name__ == "__main__":
    for name in LOADERS:
        ds = load_dataset(name)
        print(f"{name:7s} train={ds.X_train.shape} cal={ds.X_cal.shape} "
              f"test={ds.X_test.shape} positive rate={ds.y_train.mean():.3f}")
