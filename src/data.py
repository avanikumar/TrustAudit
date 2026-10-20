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
    for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype("category")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )

    return X_train, X_cal, X_test, y_train, y_cal, y_test


if __name__ == "__main__":
    X_train, X_cal, X_test, y_train, y_cal, y_test = load_data()
    print(X_train.shape, X_cal.shape, X_test.shape)
    print(y_train.mean(), y_cal.mean(), y_test.mean())