import pytest

from trustaudit.data import load_dataset
from trustaudit.explain import attributions, build_explainer
from trustaudit.model import train_model
from trustaudit.verifier import Verifier


@pytest.fixture(scope="session")
def ds():
    # Bundled with scikit-learn, so tests need no data download
    return load_dataset("cancer")


@pytest.fixture(scope="session")
def model(ds):
    return train_model(ds.X_train, ds.y_train)


@pytest.fixture(scope="session")
def explainer(model):
    return build_explainer(model)


@pytest.fixture(scope="session")
def row(ds):
    return ds.X_test.iloc[[0]]


@pytest.fixture(scope="session")
def attrs(explainer, row):
    return attributions(explainer, row)


@pytest.fixture
def verifier(ds, model):
    return Verifier(model, ds.X_train, seed=0)
