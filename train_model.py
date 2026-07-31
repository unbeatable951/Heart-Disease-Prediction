"""
train_model.py
----------------
Task 1: Data Understanding and Preprocessing
Task 2: Model Development

Loads heart.csv, explores it, splits it, trains a Logistic Regression
classifier, evaluates accuracy, and serializes the trained model (plus
the scaler and feature order) to model.pkl using Joblib.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# -------------------------------------------------------------------

# 1. Load the dataset using Pandas
df = pd.read_csv("heart.csv")

# 2. Display the first five records
print("First 5 records:")
print(df.head())

# 3. Identify numerical features and the target variable
TARGET = "target"
numerical_features = [c for c in df.columns if c != TARGET]
print("\nNumerical features:", numerical_features)
print("Target variable:", TARGET)

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Split the dataset into 80% training and 20% testing
X = df[numerical_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------------------------------------------------------
# Task 2: Model Development
# -------------------------------------------------------------------

# Scale features (helps Logistic Regression converge and perform well)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate using Accuracy Score
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel: Logistic Regression")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model (bundle model + scaler + feature order together
# so the Flask API can reproduce preprocessing exactly at inference time)
artifact = {
    "model": model,
    "scaler": scaler,
    "feature_order": numerical_features,
    "accuracy": accuracy,
}
joblib.dump(artifact, "model.pkl")
print("\nSaved trained model, scaler, and feature order to model.pkl")
