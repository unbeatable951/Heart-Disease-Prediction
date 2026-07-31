# Heart Disease Prediction — End-to-End ML Deployment

A machine learning model that predicts whether a patient is at risk of
heart disease based on clinical parameters, served through a Flask REST
API.

---

## Project Structure

```
HeartDiseaseDeployment/
├── app.py                    # Flask REST API
├── train_model.py            # Data preprocessing + model training (Tasks 1 & 2)
├── generate_sample_data.py   # (No longer used — model is trained on the real Kaggle dataset)
├── model.pkl                 # Trained model + scaler + feature order (Joblib)
├── requirements.txt          # Python dependencies
├── Procfile                  # Tells Render how to start the app (gunicorn)
├── README.md
├── heart.csv                 # Real dataset (Kaggle: johnsmith88/heart-disease-dataset)
├── templates/
│   └── index.html            # Optional demo page
└── static/                   # (Optional, currently empty)
```

---

## Task 1 & 2 — Data Preprocessing and Model Development

`train_model.py`:
- Loads `heart.csv` with Pandas and prints the first five records.
- Identifies the 13 numerical clinical features (`age`, `sex`, `cp`,
  `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`,
  `slope`, `ca`, `thal`) and the target variable (`target`).
- Checks for missing values.
- Splits the data 80/20 into train/test sets (stratified on the target).
- Scales features with `StandardScaler`.
- Trains a **Logistic Regression** classifier.
- Evaluates it with **accuracy score** and a full classification report.
- Saves the trained model, scaler, and feature order together into
  `model.pkl` using Joblib.

Run it with:

```bash
pip install -r requirements.txt
python train_model.py
```

### Model Performance

Trained on the real dataset (`heart.csv`, 1025 records):

- **Algorithm:** Logistic Regression
- **Train/Test split:** 820 / 205 records (80/20, stratified)
- **Accuracy Score:** `0.8098` (≈ 81%)

| Class                  | Precision | Recall | F1-score | Support |
|------------------------|-----------|--------|----------|---------|
| 0 (No Heart Disease)   | 0.89      | 0.70   | 0.78     | 100     |
| 1 (Heart Disease)      | 0.76      | 0.91   | 0.83     | 105     |
| **Accuracy**           |           |        | **0.81** | 205     |

The model is notably better at **recall for the positive class (0.91)**
— i.e., it catches most true heart-disease cases — while precision on
the negative class is higher (0.89), meaning it's fairly reliable when
it predicts "no disease." This recall-leaning behavior is desirable in
a healthcare screening context, where missing an at-risk patient
(false negative) is generally costlier than a false alarm.

---

## Task 3 — API Development

`app.py` exposes a Flask REST API:

| Endpoint    | Method | Description                                      |
|-------------|--------|---------------------------------------------------|
| `/`         | GET    | API info + list of required input fields          |
| `/health`   | GET    | Simple health check                                |
| `/predict`  | POST   | Accepts patient data as JSON, returns prediction   |

### Example request

```bash
curl -X POST https://YOUR-RENDER-URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

### Example response

```json
{
  "prediction": "Heart Disease Detected",
  "probability": 0.7752
}
```

Run locally:

```bash
python app.py
# API available at http://localhost:5000
```

---

## Task 4 — GitHub

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: heart disease prediction API"
git branch -M main
git remote add origin https://github.com/unbeatable951/HeartDiseaseDeployment.git
git push -u origin main
```
## Task 5 — Conclusion

The Logistic Regression model, trained on the real Kaggle heart disease
dataset, achieved an accuracy of **81%** on the held-out test set, with
strong recall (0.91) for detecting patients actually at risk of heart
disease. This shows that standard clinical parameters — such as chest
pain type, cholesterol, maximum heart rate, and exercise-induced angina
— carry meaningful predictive signal even with a relatively simple
linear model. Preprocessing steps like checking for missing values,
stratified train/test splitting, and feature scaling were essential to
getting stable, comparable results.

The main challenges faced were around deployment rather than modeling.
Locally, the Flask development server needed to be run in a dedicated
terminal window and tested from a second window, since the server
process blocks the terminal it runs in. For the actual cloud deployment,
the key considerations were packaging the model together with its
scaler and exact feature order (via Joblib) so predictions made through
the API exactly reproduce the training pipeline, and configuring Render
with the correct build (`pip install -r requirements.txt`) and start
(`gunicorn app:app`) commands so the app binds correctly to Render's
assigned port.

This project reinforced why MLOps matters in practice: a model is only
useful once it can be reproducibly trained, versioned, served through a
reliable API, and kept running in production — not just evaluated once
inside a notebook.

---

## Learning Outcomes Covered

- Built and evaluated a machine learning classification model.
- Saved and loaded a trained model using Joblib.
- Developed a REST API using Flask.
- Managed project code using GitHub.
- Deployed a machine learning application on the cloud using Render.
- Practiced core MLOps concepts: packaging, version control, deployment,
  and serving predictions through an API.
