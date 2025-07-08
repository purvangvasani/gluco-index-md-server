import os
from openai import OpenAI

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

def generate_patient_summary(prediction_result: dict) -> str:
    """
    Generates a health summary, advice, and action plan based on model output.

    Args:
        prediction_result (dict): The prediction result dict from the model.

    Returns:
        str: A UI-friendly markdown-like summary.
    """

    prediction = prediction_result.get("prediction")
    probability = prediction_result.get("probability")
    form_data = prediction_result.get("form_data", {})
    processed_reports = prediction_result.get("processed_reports", [])
    report_processed = prediction_result.get("report_processed", False)

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

    # Convert input to readable text
    input_text = f"""
        **Patient Information:**
            - Gender: {form_data.get("gender")}
            - Age: {form_data.get("age")}
            - Hypertension: {'Yes' if form_data.get("hypertension") else 'No'}
            - Heart Disease: {'Yes' if form_data.get("heart_disease") else 'No'}
            - Smoking History: {form_data.get("smoking_history")}
            - BMI: {form_data.get("bmi")}
            - HbA1c Level: {form_data.get("HbA1c_level")}%
            - Blood Glucose Level: {form_data.get("blood_glucose_level")} mg/dL

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