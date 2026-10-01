# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease from
clinical parameters, using the UCI Heart Disease dataset (303 patients, 13 features).

## Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart.csv                 # UCI Heart Disease dataset
├── src/
│   ├── preprocess.py              # Data loading & cleaning
│   ├── train.py                   # Trains & compares 6 models
│   └── evaluate.py                # EDA plots + feature importance
├── app/
│   └── app.py                     # Streamlit prediction app
├── models/                        # Saved model, scaler, figures (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
# 1. Train & compare models (saves best model to models/)
python src/train.py

# 2. Generate EDA plots + feature importance (saves to models/figures/)
python src/evaluate.py

# 3. Launch the interactive prediction app
streamlit run app/app.py
```

## Dataset

Source: UCI Machine Learning Repository — Heart Disease dataset (Cleveland subset,
via public GitHub mirror of the commonly used Kaggle version).

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia type |
| target | 1 = disease present, 0 = no disease |

## Model Results

Six models were trained and compared using accuracy, precision, recall,
F1-score, ROC-AUC, and 5-fold cross-validation:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN | 0.82 | 0.79 | 0.91 | 0.85 | **0.89** |
| Random Forest | 0.80 | 0.76 | 0.94 | 0.84 | 0.88 |
| SVM | 0.84 | 0.79 | 0.94 | 0.86 | 0.88 |
| Gradient Boosting | 0.82 | 0.81 | 0.88 | 0.84 | 0.87 |
| Logistic Regression | 0.79 | 0.76 | 0.88 | 0.82 | 0.86 |
| Decision Tree | 0.72 | 0.71 | 0.82 | 0.76 | 0.71 |

**Best model: KNN** (highest ROC-AUC), selected and saved automatically by `train.py`.
Recall was prioritized alongside ROC-AUC since missing an actual heart disease
case is more costly than a false alarm.

### Most important features (Random Forest importance)
1. Chest pain type (cp)
2. Max heart rate achieved (thalach)
3. Number of major vessels (ca)
4. Thalassemia (thal)
5. ST depression (oldpeak)

## Notes
- `train.py` is deterministic (`random_state=42`) so results are reproducible.
- Re-running `train.py` overwrites `models/best_model.pkl`.
- The Streamlit app must be run from the project root so relative paths to `models/` resolve correctly.
