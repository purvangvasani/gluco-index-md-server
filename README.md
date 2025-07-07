uvicorn main:app --reload

| Feature Name          | Type        | Description                                                       | Mandatory? |
| --------------------- | ----------- | ----------------------------------------------------------------- | ---------- |
| `gender`              | Categorical | Male / Female / Other                                             | ✅ Yes      |
| `age`                 | Numeric     | Age of the patient (e.g., 18–80+)                                 | ✅ Yes      |
| `hypertension`        | Binary      | 1 = Yes, 0 = No                                                   | ✅ Yes      |
| `heart_disease`       | Binary      | 1 = Yes, 0 = No                                                   | ✅ Yes      |
| `smoking_history`     | Categorical | \['never', 'former', 'current', 'not current', 'ever', 'No Info'] | ✅ Yes      |
| `bmi`                 | Numeric     | Body Mass Index (e.g., 18.5–40+)                                  | ✅ Yes      |
| `HbA1c_level`         | Numeric     | Hemoglobin A1c level (e.g., 4.0–12.0%)                            | ✅ Yes      |
| `blood_glucose_level` | Numeric     | Current blood glucose level (e.g., 70–400+)                       | ✅ Yes      |



calculate bmi
Weight = 58 kg
Height = 162 cm → 1.62 m

BMI = 58 / (1.62)^2 ≈ 22.1