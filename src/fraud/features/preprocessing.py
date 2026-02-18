import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["Id"])
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Risk_Flag"])
    y = df["Risk_Flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def encode_categorical(X_train, X_test):
    le_dict = {}

    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in X_train.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        le_dict[col] = le

    return X_train, X_test, le_dict
