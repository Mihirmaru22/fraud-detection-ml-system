from fastapi import FastAPI
import pandas as pd

from fraud.api.schemas import FraudInput
from fraud.pipeline.predict_pipeline import FraudPredictor


app = FastAPI()
predictor = FraudPredictor()


@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}


@app.post("/predict")
def predict(input_data: FraudInput):
    data = input_data.dict()

    # Map schema name to training name
    data["Married/Single"] = data.pop("Married_Single")

    df = pd.DataFrame([data])

    prediction = predictor.predict(df)

    return {"fraud_prediction": int(prediction[0])}
