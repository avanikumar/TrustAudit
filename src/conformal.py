import numpy as np 

from src.data import load_data  
from src.model import train_model 

def calibrate(model, X_cal, y_cal, alpha=0.1):
    proba = model.predict_proba(X_cal)                       # shape (n, 2)
    scores = 1 - proba[np.arange(len(y_cal)), y_cal.to_numpy()]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, method="higher")

def predict_sets(model, X, q):
    proba = model.predict_proba(X)
    return proba >= 1 - q        # boolean array (n, 2): is each class in the set?

def evaluate_sets(sets, y):
    y = y.to_numpy()
    covered = sets[np.arange(len(y)), y]      # True if the true class is in the set
    size = sets.sum(axis=1)                   # 0, 1 or 2 classes in each set
    return covered.mean(), size.mean(), (size == 2).mean()

if __name__ == "__main__":
    X_train, X_cal, X_test, y_train, y_cal, y_test = load_data()
    model = train_model(X_train, y_train)
    for alpha in [0.05, 0.1, 0.2]:
        q = calibrate(model, X_cal, y_cal, alpha)
        sets = predict_sets(model, X_test, q)
        cov, size, both = evaluate_sets(sets, y_test)
        print(f"alpha={alpha:<5} target={1-alpha:.2f} coverage={cov:.3f} avg_size={size:.2f} both={both:.2f}")
        