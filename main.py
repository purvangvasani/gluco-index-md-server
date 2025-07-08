# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from fastapi import Form, File, UploadFile
from pathlib import Path
import uuid
from dotenv import load_dotenv
from data_extract import DocumentProcessor
import os
# Load trained pipeline (includes preprocessing)
# model = joblib.load("diabetes_model_max_recall.pkl")
model = joblib.load("lightgbm_diabetes_model.pkl")
# Load environment variables
load_dotenv()

class ReportData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int
    report_text: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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

@app.post("/submit-report")
async def submit_report(
    gender: str = Form(...),
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    smoking_history: str = Form(...),
    bmi: float = Form(...),
    HbA1c_level: float = Form(...),
    blood_glucose_level: int = Form(...),
    report_file: UploadFile = File(None)
):
    try:
        extracted_data = None
        
        # Process the uploaded file if present
        if report_file and report_file.filename:
            # Save the uploaded file
            file_ext = Path(report_file.filename).suffix
            file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_ext}")
            
            with open(file_path, "wb") as buffer:
                buffer.write(await report_file.read())
            
            # Process the document
            try:
                processor = DocumentProcessor()
                result = processor.process_document(file_path)
                
                if "error" in result:
                    raise HTTPException(status_code=400, detail=f"Error processing document: {result['error']}")
                
                extracted_data = result.get("extracted_data", {})
                
                # Update form data with extracted values if available
                if "hba1c" in extracted_data and "value" in extracted_data["hba1c"]:
                    HbA1c_level = float(extracted_data["hba1c"]["value"])
                if "glucose" in extracted_data and "value" in extracted_data["glucose"]:
                    blood_glucose_level = int(float(extracted_data["glucose"]["value"]))
            
            except Exception as e:
                # If document processing fails, continue with form data
                print(f"Warning: Could not process document: {str(e)}")
        
        # Create a dictionary with all form data
        form_data = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": HbA1c_level,
            "blood_glucose_level": blood_glucose_level,
            "extracted_data": extracted_data
        }
        
        # Make prediction (using your existing prediction logic)
        input_df = pd.DataFrame([form_data])
        
        # Load model and make prediction
        model = joblib.load('diabetes_model.pkl')
        prediction = model.predict(input_df)
        
        # Return the results
        return {
            "prediction": int(prediction[0]),
            "form_data": form_data,
            "report_processed": extracted_data is not None
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))