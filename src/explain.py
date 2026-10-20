import numpy as np 
import shap 

from src.data import load_data 
from src.model import train_model 

def build_explainer(model):
    return shap.TreeExplainer(model)

def get_top_features(explainer, X_row, k=5):
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):      # some SHAP versions return one array per class
        sv = sv[1]
    sv = np.asarray(sv)[0]        # SHAP values for the single row

    order = np.argsort(-np.abs(sv))[:k]   # indices of the k largest effects
    results = []
    for i in order:
        results.append({
            "feature": X_row.columns[i],
            "value": X_row.iloc[0, i],
            "shap": float(sv[i]),
            "direction": "increases risk" if sv[i] > 0 else "decreases risk",
        })
    return results

if __name__ == "__main__":
    X_train, X_cal, X_test, y_train, y_cal, y_test = load_data()
    model = train_model(X_train, y_train)
    explainer = build_explainer(model)
    for i in range(3):
        row = X_test.iloc[[i]]
        print(f"\nCustomer {i}: actual churn = {y_test.iloc[i]}")
        for f in get_top_features(explainer, row):
            print(f"  {f['feature']:<18} = {str(f['value']):<18} shap={f['shap']:+.3f}  ({f['direction']})")
            
            
            