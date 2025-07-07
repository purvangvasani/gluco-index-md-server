# train_diabetes_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')
print("Initial Data Shape:", df.shape)

# Optional: remove duplicates
df = df.drop_duplicates()

# 2. Define features and target
X = df.drop(columns=['diabetes'])
y = df['diabetes']

# 3. Define numeric and categorical columns
categorical_cols = ['gender', 'smoking_history']
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# 4. Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# 5. Build pipeline
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42))
        ],
        voting='soft'
    ))
])

# 6. Hyperparameter tuning
param_grid = {
    'classifier__rf__n_estimators': [100, 200],
    'classifier__rf__max_depth': [5, 10],
    'classifier__gb__learning_rate': [0.05, 0.1],
    'classifier__gb__n_estimators': [100, 200]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    model_pipeline,
    param_grid=param_grid,
    scoring='recall',  # Focus on recall
    cv=cv,
    verbose=2,
    n_jobs=-1
)

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 8. Fit the model
grid.fit(X_train, y_train)

# 9. Evaluate
y_pred = grid.predict(X_test)

print("\nBest Hyperparameters:", grid.best_params_)
print("Recall-focused Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Save the model
joblib.dump(grid.best_estimator_, 'diabetes_model_max_recall.pkl')
print("\nModel saved to 'diabetes_model_max_recall.pkl'")
