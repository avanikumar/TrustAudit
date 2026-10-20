import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="data/telco.csv", seed=42):
    # 1. read csv
    # 2. drop customerID
    # 3. fix TotalCharges
    # 4. make Churn 0/1
    # 5. object columns -> category
    # 6. split into X and y
    # 7. two-step 60/20/20 split with stratify
    # 8. return X_train, X_cal, X_test, y_train, y_cal, y_test
    
    df=pd.read_csv(path)
    df=df.drop(columns=["customerID"])
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    df["Churn"]=(df["Churn"] == "Yes").astype(int)
    print(df.shape, df["Churn"].mean())


if __name__ == "__main__":
    X_train, X_cal, X_test, y_train, y_cal, y_test = load_data()
    print(X_train.shape, X_cal.shape, X_test.shape)
    print(y_train.mean(), y_cal.mean(), y_test.mean())