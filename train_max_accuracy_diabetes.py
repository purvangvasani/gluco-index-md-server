# train_max_accuracy_diabetes.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 1. Load & Clean Data
df = pd.read_csv('diabetes_prediction_dataset.csv')
df = df.drop_duplicates()
print("Data shape:", df.shape)

# 2. Define Features & Target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# 3. Preprocessing
# - One-hot encode gender and smoking_history
# - Scale numeric features
cat_features = ['gender', 'smoking_history']
num_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

# 4. Build ML Pipeline with SMOTE + Voting Classifier
clf = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('vote', VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42))
    ], voting='soft'))
])

# 5. Hyperparameter Tuning
param_grid = {
    'vote__rf__max_depth': [5, 10],
    'vote__gb__learning_rate': [0.05, 0.1],
    'vote__gb__max_depth': [3, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(clf, param_grid, cv=cv, scoring='recall', n_jobs=-1, verbose=2)
grid.fit(X, y)

print("Best params:", grid.best_params_)

# 6. Evaluation
y_pred = grid.predict(X)
print(classification_report(y, y_pred))
print("Confusion matrix:\n", confusion_matrix(y, y_pred))

# 7. Save Model
joblib.dump(grid.best_estimator_, 'max_accuracy_diabetes.pkl')
print("Model saved!")
