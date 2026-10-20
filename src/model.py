from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from src.data import load_data
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
    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, proba > 0.5)
    return auc, acc


if __name__ == "__main__":
    X_train, X_cal, X_test, y_train, y_cal, y_test = load_data()
    model = train_model(X_train, y_train)
    cal_auc, cal_acc = evaluate(model, X_cal, y_cal)
    test_auc, test_acc = evaluate(model, X_test, y_test)
    print(f"cal  AUC={cal_auc:.3f} acc={cal_acc:.3f}")
    print(f"test AUC={test_auc:.3f} acc={test_acc:.3f}")
    
    