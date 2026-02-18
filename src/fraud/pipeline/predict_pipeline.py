import joblib
import pandas as pd
import os


class FraudPredictor:
    def __init__(self):
        BASE_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../")
        )

        model_path = os.path.join(BASE_DIR, "artifacts/model.pkl")
        encoder_path = os.path.join(BASE_DIR, "artifacts/encoders.pkl")

        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoder_path)

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        # Validate columns
        for col in self.encoders.keys():
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Apply encoders
        for col, encoder in self.encoders.items():
            try:
                data[col] = encoder.transform(data[col])
            except ValueError:
                raise ValueError(f"Unknown category in column {col}")

        return data

    def predict(self, data: pd.DataFrame):
        processed = self.preprocess(data)
        return self.model.predict(processed)


if __name__ == "__main__":
    sample = {
        "Income": 500000,
        "Age": 45,
        "Experience": 10,
        "Married/Single": "single",
        "House_Ownership": "rented",
        "Car_Ownership": "no",
        "Profession": "Engineer",
        "CITY": "Mumbai",
        "STATE": "Maharashtra",
        "CURRENT_JOB_YRS": 5,
        "CURRENT_HOUSE_YRS": 3
    }

    df = pd.DataFrame([sample])

    predictor = FraudPredictor()
    print("Prediction:", predictor.predict(df))
