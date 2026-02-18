from fraud.features.preprocessing import load_data, split_data, encode_categorical
from fraud.models.train import train_models, select_best_model


def run_training():
    path = "data/raw/LoanPrediction.csv"

    # 1. Load data
    df = load_data(path)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Encode
    X_train, X_test, encoders = encode_categorical(X_train, X_test)

    # 4. Train models
    trained_models = train_models(X_train, y_train)

    # 5. Select best model
    best_model, best_name = select_best_model(
        trained_models, X_test, y_test
    )

    print(f"Best model selected: {best_name}")


if __name__ == "__main__":
    run_training()
