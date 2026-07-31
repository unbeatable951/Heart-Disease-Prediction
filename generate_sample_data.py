"""
Generates a SYNTHETIC heart.csv that mirrors the column schema of the
Kaggle 'johnsmith88/heart-disease-dataset' (the standard UCI Cleveland
heart-disease feature set: age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal, target).

IMPORTANT: This is fake data used only so the pipeline can be built and
tested end-to-end without network access to Kaggle. Before submitting,
download the REAL dataset from:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
and overwrite heart.csv with it.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

age = np.random.randint(29, 78, n)
sex = np.random.randint(0, 2, n)
cp = np.random.randint(0, 4, n)
trestbps = np.random.randint(94, 201, n)
chol = np.random.randint(126, 565, n)
fbs = np.random.randint(0, 2, n)
restecg = np.random.randint(0, 3, n)
thalach = np.random.randint(71, 203, n)
exang = np.random.randint(0, 2, n)
oldpeak = np.round(np.random.uniform(0, 6.2, n), 1)
slope = np.random.randint(0, 3, n)
ca = np.random.randint(0, 5, n)
thal = np.random.randint(0, 4, n)

# Construct a target with some signal so the model isn't pure noise
risk_score = (
    (age > 54).astype(int) +
    (chol > 240).astype(int) +
    (thalach < 140).astype(int) +
    exang +
    (oldpeak > 1.5).astype(int) +
    (cp == 0).astype(int)
)
prob = 1 / (1 + np.exp(-(risk_score - 3)))
target = (np.random.uniform(0, 1, n) < prob).astype(int)

df = pd.DataFrame({
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    "target": target
})

df.to_csv("heart.csv", index=False)
print("Synthetic heart.csv written:", df.shape)
print(df["target"].value_counts())
