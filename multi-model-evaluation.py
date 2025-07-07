# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline

# # Optional: install external libs
# # pip install xgboost lightgbm imbalanced-learn

# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier

# # ------------------------------
# # 1. Load and Prepare Data
# # ------------------------------
# df = pd.read_csv("diabetes_prediction_dataset.csv")
# df = df.drop_duplicates()

# X = df.drop(columns=['diabetes'])
# y = df['diabetes']

# categorical_cols = ['gender', 'smoking_history']
# numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# preprocessor = ColumnTransformer([
#     ('num', StandardScaler(), numeric_cols),
#     ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
# ])

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# # ------------------------------
# # 2. Define Models to Evaluate
# # ------------------------------
# models = {
#     "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
#     "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
#     "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
#     "SVM": SVC(probability=True, class_weight='balanced'),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, learning_rate=0.1),
#     "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1)
# }

# # ------------------------------
# # 3. Train & Evaluate
# # ------------------------------
# results = []

# for name, clf in models.items():
#     print(f"\n🔍 Training: {name}")
#     pipeline = ImbPipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('smote', SMOTE(random_state=42)),
#         ('classifier', clf)
#     ])

#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print("Classification Report:\n", classification_report(y_test, y_pred))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#     print("Accuracy:", acc)

#     results.append({
#         "Model": name,
#         "Accuracy": round(acc, 4),
#         "Recall (Diabetic)": round(recall, 4),
#         "F1 (Diabetic)": round(f1, 4)
#     })

#     joblib.dump(pipeline, f"{name.lower()}_diabetes_model.pkl")
#     print(f"✅ Saved: {name.lower()}_diabetes_model.pkl")

# # ------------------------------
# # 4. Print Summary Table
# # ------------------------------
# print("\n📊 Summary Comparison:")
# summary_df = pd.DataFrame(results)
# print(summary_df.sort_values(by="Recall (Diabetic)", ascending=False).to_string(index=False))


import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ------------------------------
# 1. Load and Prepare Data
# ------------------------------
df = pd.read_csv("diabetes_prediction_dataset.csv").drop_duplicates()

X = df.drop(columns=['diabetes'])
y = df['diabetes']

categorical_cols = ['gender', 'smoking_history']
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ------------------------------
# 2. Define Models to Evaluate
# ------------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    # "SVM": SVC(probability=True, class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, learning_rate=0.1),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.1),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = []
best_xgb_pipeline = None  # to save for SHAP

for name, clf in models.items():
    print(f"\n🔍 Training: {name}")
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', clf)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", acc)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Recall (Diabetic)": round(recall, 4),
        "F1 (Diabetic)": round(f1, 4)
    })

    joblib.dump(pipeline, f"{name.lower()}_diabetes_model.pkl")
    print(f"✅ Saved: {name.lower()}_diabetes_model.pkl")

    if name == "XGBoost":
        best_xgb_pipeline = pipeline

# ------------------------------
# 3. Print Summary Table
# ------------------------------
print("\n📊 Summary Comparison:")
summary_df = pd.DataFrame(results)
print(summary_df.sort_values(by="Recall (Diabetic)", ascending=False).to_string(index=False))

# ------------------------------
# 4. SHAP Explainability (XGBoost)
# ------------------------------
if best_xgb_pipeline:
    print("\n🧠 SHAP Explainability for XGBoost...")

    # Transform test data manually (required for SHAP)
    X_test_transformed = best_xgb_pipeline.named_steps['preprocessor'].transform(X_test)
    xgb_model = best_xgb_pipeline.named_steps['classifier']

    # Initialize SHAP explainer
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_test_transformed)

    # Summary plot (top features)
    shap.summary_plot(shap_values, X_test_transformed, feature_names=
        best_xgb_pipeline.named_steps['preprocessor'].get_feature_names_out(), show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    print("📸 SHAP summary plot saved as: shap_summary_plot.png")
