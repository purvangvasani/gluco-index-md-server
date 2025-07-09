# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import Form, File, UploadFile, status, Path as FPath
from pathlib import Path
import uuid
from dotenv import load_dotenv
from data_extract import DocumentProcessor
import os
from summary import generate_patient_summary
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

def get_upload_path(filename: str) -> Path:
    """Get the full path to the uploaded file."""
    return Path(UPLOAD_DIR) / filename

@app.delete("/report/{filename}", status_code=status.HTTP_200_OK)
async def delete_report(
    filename: str = FPath(..., description="Name of the file to delete")
):
    """
    Delete a report file from the server.
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        dict: A dictionary containing the status of the operation
    """
    try:
        file_path = get_upload_path(filename)
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' not found"
            )
            
        # Check if the path is actually a file (not a directory)
        if not file_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{filename}' is not a valid file"
            )
            
        # Delete the file
        file_path.unlink()
        
        return {
            "status": "success",
            "message": f"File '{filename}' deleted successfully",
            "filename": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting the file: {str(e)}"
        )

@app.post("/upload-report", status_code=status.HTTP_201_CREATED)
async def upload_report(file: UploadFile = File(...)):
    """
    Upload a report file to the server.
    
    Args:
        file: The file to be uploaded
        
    Returns:
        dict: A dictionary containing the filename and success status
    """
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate a unique filename to prevent overwrites
        file_ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        return {
            "status": "success",
            "filename": filename,
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while uploading the file: {str(e)}"
        )

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

class ReportReference(BaseModel):
    report_references: List[str] = []

class ReportData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: Optional[float] = None
    blood_glucose_level: Optional[int] = None
    report_references: List[str] = []

@app.post("/submit-report")
async def submit_report(data: ReportData):
    try:
        extracted_data = {}
        processed_files = []

        # Process each referenced report
        for filename in data.report_references:
            file_path = get_upload_path(filename)
            if not file_path.exists() or not file_path.is_file():
                print(f"Warning: Report file '{filename}' not found")
                continue

            try:
                processor = DocumentProcessor()
                result = processor.process_document(str(file_path))

                if "error" in result:
                    print(f"Warning: Error processing document {filename}: {result['error']}")
                    continue

                # Extract relevant Test_Results
                test_results = result.get("test_results", {})
                if test_results:
                    if "HbA1c" in test_results and "value" in test_results["HbA1c"]:
                        hba1c_level = float(test_results["HbA1c"]["value"])

                    if "Glucose" in test_results and "value" in test_results["Glucose"]:
                        glucose_level = int(float(test_results["Glucose"]["value"]))
                
                if test_results:
                    extracted_data.update(test_results)
                    processed_files.append(filename)

            except Exception as e:
                print(f"Error processing document {filename}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Fallback values from form
        hba1c_level = data.HbA1c_level
        glucose_level = data.blood_glucose_level
        # print("extracted_data: ", extracted_data)
        # Try extracting from processed data if available
        if extracted_data:
            # print(f"Extracted keys: {list(extracted_data.keys())}")

            hba1c_info = extracted_data.get("HbA1c")
            glucose_info = extracted_data.get("Glucose")

            if hba1c_info and isinstance(hba1c_info, dict):
                try:
                    hba1c_level = float(hba1c_info.get("value", hba1c_level))
                except Exception:
                    pass

            if glucose_info and isinstance(glucose_info, dict):
                try:
                    glucose_level = int(float(glucose_info.get("value", glucose_level)))
                except Exception:
                    pass

        # Final input for prediction
        form_data = {
            "gender": data.gender,
            "age": data.age,
            "hypertension": data.hypertension,
            "heart_disease": data.heart_disease,
            "smoking_history": data.smoking_history,
            "bmi": data.bmi,
            "HbA1c_level": hba1c_level,
            "blood_glucose_level": glucose_level
        }

        # print("Prepared form data:", form_data)
        # result_data = predict(DiabetesInput(**form_data))
        # print("Result data: ", result_data)
        # Predict
        input_df = pd.DataFrame([form_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        # print("Probability: ", {
        #     "prediction": int(prediction),
        #     "probability": round(probability * 100, 2),
        #     "form_data": form_data,
        #     "report_processed": bool(extracted_data),
        #     "processed_reports": processed_files
        # })
        summary = generate_patient_summary(form_data)
        for filename in data.report_references:
            delete_report(filename)
        
        return {
            "prediction": int(prediction),
            "probability": round(probability * 100, 2),
            "form_data": form_data,
            "report_processed": bool(extracted_data),
            "processed_reports": processed_files,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")
