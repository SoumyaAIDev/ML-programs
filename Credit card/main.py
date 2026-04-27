from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Credit Compliance API Running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[
        data["interest_rate"],
        data["late_fee"],
        data["annual_fee"],
        data["billing_cycle"],
        data["min_payment"],
        data["disclosure"]
    ]])

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]

    result = "Compliant" if prediction == 1 else "Non-Compliant"

    return {
        "prediction": result,
        "confidence": float(prob)
    }