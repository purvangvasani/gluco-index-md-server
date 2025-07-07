# shap_explainer.py

import shap
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset used during training
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Define features and target
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# Train/test split for SHAP visualizations
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load the trained model pipeline (includes preprocessing and classifier)
model = joblib.load("diabetes_model_max_recall.pkl")

# 1. SHAP Explainer using TreeExplainer (for tree-based models)
# Since VotingClassifier includes tree models, this is appropriate

print("Creating SHAP explainer...")
explainer = shap.Explainer(model.named_steps["classifier"])

# Preprocess the test data using the pipeline
X_transformed = model.named_steps["preprocessor"].transform(X_test)

# Get SHAP values
print("Calculating SHAP values...")
shap_values = explainer(X_transformed)

# 2. Global Feature Importance
shap.plots.beeswarm(shap_values)

# 3. Individual Explanation (1st sample)
shap.plots.waterfall(shap_values[0])

# Optional: visualize summary bar chart
shap.plots.bar(shap_values)

# For interactive notebook:
# shap.initjs()
# shap.force_plot(explainer.expected_value[1], shap_values[0].values, feature_names=...) 
