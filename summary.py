import os
from openai import OpenAI

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

def generate_patient_summary(prediction: int, probability: float, data, processed_reports=None, report_processed=False) -> str:
    """
    Generates a health summary, advice, and action plan based on model output.

    Args:
        prediction (int): 0 or 1 indicating diabetes risk.
        probability (float): Confidence percentage.
        data: Pydantic model or object containing form input fields.
        processed_reports (list): List of processed report names (optional).
        report_processed (bool): Flag to indicate if any reports were processed.

    Returns:
        str: A UI-friendly markdown-like summary.
    """
    processed_reports = processed_reports or []

    system_prompt = (
        "You are a compassionate AI health assistant named GlucoIndex-md. "
        "Generate a UI-friendly health summary using markdown-like formatting. "
        "Use **bold** for key terms, and emojis like ✅, ❌, 🩺, 💡, ⚠️, 📋, etc. for visual cues. "
        "Break down the response into sections:\n"
        "1. 🩺 **Health Summary**\n"
        "2. ⚠️ **Risks**\n"
        "3. 📋 **Do's and Don'ts**\n"
        "4. 💡 **Lifestyle Suggestions**\n"
        "5. 🕐 **When to See a Doctor**\n"
        "6. 💬 **Final Encouragement**\n"
        "Make it warm, simple, and supportive for a patient."
    )

    input_text = f"""
        **Patient Information:**
            - Gender: {data.gender}
            - Age: {data.age}
            - Hypertension: {'Yes' if data.hypertension else 'No'}
            - Heart Disease: {'Yes' if data.heart_disease else 'No'}
            - Smoking History: {data.smoking_history}
            - BMI: {data.bmi}
            - HbA1c Level: {data.HbA1c_level}%
            - Blood Glucose Level: {data.blood_glucose_level} mg/dL

        **Prediction Outcome:**
            - Diabetes Risk: {"🟥 **Positive**" if prediction else "🟩 **Negative**"}
            - Probability: **{probability}%**

        Reports Analyzed: {', '.join(processed_reports) if report_processed else "None"}
    """

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
        temperature=0.7,
        max_tokens=900,
    )

    return response.choices[0].message.content.strip()
    
# if __name__ == "__main__":
#     # Replace this with the actual model result you got from your API
#     model_result = {
#         'prediction': 1,
#         'probability': 100.0,
#         'form_data': {
#             'gender': 'male',
#             'age': 27.0,
#             'hypertension': 0,
#             'heart_disease': 0,
#             'smoking_history': 'never',
#             'bmi': 23.44,
#             'HbA1c_level': 11.4,
#             'blood_glucose_level': 280
#         },
#         'report_processed': True,
#         'processed_reports': ['dc5f445b-6594-4b56-9ac2-4dd1db5357e1.jpg']
#     }

#     summary = generate_patient_summary(model_result)
#     print("\n=== Patient Summary ===\n")
#     print(summary)