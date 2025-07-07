# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load trained pipeline (includes preprocessing)
# model = joblib.load("diabetes_model_max_recall.pkl")
model = joblib.load("lightgbm_diabetes_model.pkl")

# Define input schema
class DiabetesInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Diabetes Prediction API"}

@app.post("/predict")
def predict(data: DiabetesInput):
    # Step 1: Build dictionary of input
    input_dict = {
        'gender': [data.gender],
        'age': [data.age],
        'hypertension': [data.hypertension],
        'heart_disease': [data.heart_disease],
        'smoking_history': [data.smoking_history],
        'bmi': [data.bmi],
        'HbA1c_level': [data.HbA1c_level],
        'blood_glucose_level': [data.blood_glucose_level]
    }

    # Step 2: Convert to pandas DataFrame
    input_df = pd.DataFrame(input_dict)

    # Step 3: Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(probability * 100, 2)
    }
