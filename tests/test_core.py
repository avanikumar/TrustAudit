import numpy as np

from trustaudit.conformal import calibrate, describe_set, evaluate_sets, predict_sets
from trustaudit.data import load_dataset


def test_split_is_60_20_20_and_stratified(ds):
    n = len(ds.X_train) + len(ds.X_cal) + len(ds.X_test)
    assert abs(len(ds.X_train) / n - 0.6) < 0.01
    assert abs(len(ds.X_test) / n - 0.2) < 0.01
    assert abs(ds.y_train.mean() - ds.y_test.mean()) < 0.03


def test_unknown_dataset_raises():
    import pytest
    with pytest.raises(ValueError):
        load_dataset("nope")


def test_attributions_sorted_and_ranked(attrs):
    mags = [abs(a["shap"]) for a in attrs]
    assert mags == sorted(mags, reverse=True)
    assert [a["rank"] for a in attrs] == list(range(1, len(attrs) + 1))
    assert abs(sum(a["share"] for a in attrs) - 1) < 1e-6


def test_conformal_coverage_near_target(ds, model):
    q = calibrate(model, ds.X_cal, ds.y_cal, alpha=0.1)
    cov, size, _ = evaluate_sets(predict_sets(model, ds.X_test, q), ds.y_test)
    assert cov >= 0.85          # small test set, so allow some slack below 0.90
    assert 0 <= size <= 2


def test_describe_set():
    assert describe_set(np.array([False, True]), "churn") == "confident: churn"
    assert describe_set(np.array([True, False]), "churn") == "confident: no churn"
    assert describe_set(np.array([True, True]), "churn").startswith("uncertain")
    assert describe_set(np.array([False, False]), "churn").startswith("abstain")
