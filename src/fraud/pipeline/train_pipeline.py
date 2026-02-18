from fraud.features.preprocessing import load_data, split_data, encode_categorical
from fraud.models.train import train_models, select_best_model
import joblib
import os
import json


def run_training():
    path = "data/raw/LoanPrediction.csv"

    # Load
    df = load_data(path)

    # Split
    X_train, X_test, y_train, y_test = split_data(df)

    # Encode
    X_train, X_test, encoders = encode_categorical(X_train, X_test)

    # Save feature order BEFORE training
    feature_order = list(X_train.columns)

    # Train
    trained_models = train_models(X_train, y_train)

    # Select
    best_model, best_name = select_best_model(
        trained_models, X_test, y_test
    )

    print(f"Best model selected: {best_name}")

    # ---------- Save artifacts ----------
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(best_model, "artifacts/model.pkl")
    joblib.dump(encoders, "artifacts/encoders.pkl")

    # ---------- Save metadata ----------
    metadata = {
        "best_model": best_name,
        "feature_order": feature_order
    }

    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Artifacts saved successfully.")


if __name__ == "__main__":
    run_training()
