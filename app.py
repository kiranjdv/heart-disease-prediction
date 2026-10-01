"""
app.py
Production-Grade Flask Web Application for Heart Disease Prediction & Cardiology Intelligence.
Provides REST APIs and modern web dashboard matching medical UI standards.
"""

import sys
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, Response

from src.pdf_generator import generate_clinical_pdf

app = Flask(__name__)

# Base directories resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")

# Load ML artifacts
model = None
scaler = None
feature_columns = []

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(COLUMNS_PATH):
        feature_columns = joblib.load(COLUMNS_PATH)
except Exception as e:
    print(f"Error loading model artifacts: {e}")

# Patient Presets
PRESET_PATIENTS = {
    "eleanor": {
        "id": "eleanor",
        "name": "Eleanor Pena",
        "dob": "16 apr 2003",
        "age": 23,
        "sex": 0,
        "doctor": "Marry Wroons",
        "diagnosis": "Normal Cardiac Rhythm",
        "weight": "55 kg",
        "height": "175 cm",
        "vitals": {
            "age": 23, "sex": 0, "cp": 1, "trestbps": 120, "chol": 178,
            "fbs": 0, "restecg": 0, "thalach": 168, "exang": 0, "oldpeak": 0.2,
            "slope": 2, "ca": 0, "thal": 2,
        },
    },
    "william": {
        "id": "william",
        "name": "William Davis",
        "dob": "08 sep 1978",
        "age": 48,
        "sex": 1,
        "doctor": "Ethan Sanchez",
        "diagnosis": "Hypertension Stage 1",
        "weight": "82 kg",
        "height": "180 cm",
        "vitals": {
            "age": 48, "sex": 1, "cp": 2, "trestbps": 138, "chol": 235,
            "fbs": 0, "restecg": 1, "thalach": 142, "exang": 0, "oldpeak": 1.2,
            "slope": 1, "ca": 1, "thal": 2,
        },
    },
    "arthur": {
        "id": "arthur",
        "name": "Arthur Vance",
        "dob": "12 nov 1957",
        "age": 69,
        "sex": 1,
        "doctor": "Clara Oswald",
        "diagnosis": "Suspected Coronary Ischemia",
        "weight": "79 kg",
        "height": "172 cm",
        "vitals": {
            "age": 69, "sex": 1, "cp": 0, "trestbps": 162, "chol": 288,
            "fbs": 1, "restecg": 2, "thalach": 105, "exang": 1, "oldpeak": 2.8,
            "slope": 1, "ca": 3, "thal": 3,
        },
    },
}


# Realistic names pool for dataset patient indexing (303 records)
MALE_NAMES = [
    "James Smith", "John Johnson", "Robert Williams", "Michael Brown", "William Jones",
    "David Garcia", "Richard Miller", "Joseph Davis", "Thomas Rodriguez", "Charles Martinez",
    "Christopher Hernandez", "Daniel Lopez", "Matthew Gonzalez", "Anthony Wilson", "Mark Anderson",
    "Donald Thomas", "Steven Taylor", "Paul Moore", "Andrew Jackson", "Joshua Martin",
    "Kevin White", "Brian Harris", "George Clark", "Edward Lewis", "Ronald Robinson"
]
FEMALE_NAMES = [
    "Emma Smith", "Olivia Johnson", "Sophia Williams", "Isabella Brown", "Mia Jones",
    "Charlotte Garcia", "Amelia Miller", "Harper Davis", "Evelyn Rodriguez", "Abigail Martinez",
    "Emily Hernandez", "Elizabeth Lopez", "Sofia Gonzalez", "Avery Wilson", "Ella Anderson",
    "Scarlett Thomas", "Grace Taylor", "Chloe Moore", "Victoria Jackson", "Riley Martin",
    "Lily White", "Hannah Harris", "Zoey Clark", "Nora Lewis", "Layla Robinson"
]


def get_dataset_patient(patient_id):
    """Retrieves an anonymized patient from heart.csv by assigned Patient ID (PID-001 to PID-303)."""
    if not os.path.exists(DATA_PATH):
        return None

    clean_id = patient_id.upper().strip()
    idx = -1
    if clean_id.startswith("PID-"):
        try:
            idx = int(clean_id.replace("PID-", "")) - 1
        except ValueError:
            return None
    elif clean_id.isdigit():
        idx = int(clean_id) - 1

    if idx < 0:
        return None

    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]

        if idx >= len(df):
            return None

        row = df.iloc[idx]
        is_male = int(row["sex"]) == 1
        pool = MALE_NAMES if is_male else FEMALE_NAMES
        assigned_name = pool[idx % len(pool)]

        vitals = {col: float(row[col]) for col in feature_columns if col in row}

        # Clinical summary note based on biomarkers
        if vitals.get("oldpeak", 0) >= 2.0 or vitals.get("ca", 0) >= 2:
            diag = "Ischemic Risk Indicators Detected"
        elif vitals.get("trestbps", 0) >= 140 or vitals.get("chol", 0) >= 240:
            diag = "Hypertension / Metabolic Elevation"
        else:
            diag = "Hemodynamically Stable Cohort Record"

        return {
            "id": f"PID-{idx+1:03d}",
            "name": f"{assigned_name} (PID-{idx+1:03d})",
            "dob": f"Dataset Record #{idx+1}",
            "age": int(row["age"]),
            "sex": int(row["sex"]),
            "doctor": "Marry Wroons",
            "diagnosis": diag,
            "weight": "70 kg",
            "height": "172 cm",
            "vitals": vitals,
        }
    except Exception as e:
        print(f"Error resolving dataset patient: {e}")
        return None


def resolve_patient(patient_id):
    """Resolves patient by ID: checks dataset PID (e.g. PID-042). Defaults to PID-001."""
    if not patient_id:
        return get_dataset_patient("PID-001")
    ds_patient = get_dataset_patient(patient_id)
    if ds_patient:
        return ds_patient
    return get_dataset_patient("PID-001")


def perform_inference(vitals_dict):
    """Helper to transform and predict from a 13-feature dict."""
    if model is None or scaler is None or not feature_columns:
        return 0, 0.1

    input_df = pd.DataFrame([vitals_dict])[feature_columns]
    scaled = scaler.transform(input_df)
    pred = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1])
    return pred, prob


def get_clinical_recommendations(pred, prob, vitals):
    """Generates patient-tailored clinical protocol and action plan."""
    trestbps = int(vitals.get("trestbps", 120))
    chol = int(vitals.get("chol", 180))
    ca = int(vitals.get("ca", 0))

    if pred == 1:
        recs = [
            {
                "title": "Coronary Angiography & Stress Echo",
                "sub": f"Urgent secondary diagnostic evaluation recommended within 7–10 days ({ca} vessel risk indicators)",
                "badge": "Urgent Action",
                "type": "danger",
                "icon": "alert",
            },
            {
                "title": "Lipid & BP Management Protocol",
                "sub": f"Target LDL < 70 mg/dl & BP < 130/80 mmHg (Current BP: {trestbps} mmHg, Chol: {chol} mg/dl)",
                "badge": "Rx Protocol",
                "type": "warning",
                "icon": "pill",
            },
            {
                "title": "48-Hour Holter Telemetry",
                "sub": "Continuous ambulatory ECG monitoring to detect silent ischemic ST deviations",
                "badge": "In 2 Weeks",
                "type": "stress",
                "icon": "activity",
            },
        ]
    else:
        recs = [
            {
                "title": "Preventive Cardiology Review",
                "sub": "Routine annual 12-lead resting ECG and comprehensive metabolic profile",
                "badge": "Annual Plan",
                "type": "success",
                "icon": "calendar",
            },
            {
                "title": "Cardioprotective Lifestyle Program",
                "sub": "Target >= 150 mins/week moderate aerobic exercise + low-sodium DASH diet",
                "badge": "Lifestyle",
                "type": "med",
                "icon": "heart",
            },
            {
                "title": "Repeat Lipid Profile Screening",
                "sub": f"Monitor total cholesterol (Current: {chol} mg/dl) and maintain optimal HDL/LDL ratio",
                "badge": "In 6 Months",
                "type": "info",
                "icon": "check",
            },
        ]
    return recs


# ---------------------------------------------------------
# View Routes
# ---------------------------------------------------------
@app.route("/")
@app.route("/dashboard")
def dashboard():
    patient_id = request.args.get("patient", "PID-001")
    patient = resolve_patient(patient_id)

    # Compute live inference for the patient
    pred, prob = perform_inference(patient["vitals"])
    recommendations = get_clinical_recommendations(pred, prob, patient["vitals"])

    return render_template(
        "index.html",
        active_page="dashboard",
        patient=patient,
        pred=pred,
        prob=prob,
        vitals=patient["vitals"],
        recommendations=recommendations,
    )


@app.route("/results")
@app.route("/predictor")
def predictor():
    patient_id = request.args.get("patient", "PID-001")
    patient = resolve_patient(patient_id)
    pred, prob = perform_inference(patient["vitals"])
    return render_template(
        "predictor.html",
        active_page="results",
        patient=patient,
        pred=pred,
        prob=prob,
        vitals=patient["vitals"],
    )


@app.route("/analytics")
@app.route("/labs")
def analytics():
    return render_template("analytics.html", active_page="analytics")


@app.route("/leaderboard")
def leaderboard():
    comparison_data = []
    if os.path.exists(COMPARISON_PATH):
        df = pd.read_csv(COMPARISON_PATH)
        comparison_data = df.to_dict(orient="records")
    return render_template("leaderboard.html", active_page="leaderboard", models=comparison_data)


@app.route("/batch")
def batch():
    return render_template("batch.html", active_page="batch")


# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    try:
        clean_vitals = {col: float(data.get(col, 0)) for col in feature_columns}
        pred, prob = perform_inference(clean_vitals)

        # Calculate clinical risk flags
        triggers = []
        if clean_vitals.get("trestbps", 0) >= 140:
            triggers.append(f"Hypertension Stage 2 ({int(clean_vitals['trestbps'])} mmHg)")
        if clean_vitals.get("chol", 0) >= 240:
            triggers.append(f"Hypercholesterolemia ({int(clean_vitals['chol'])} mg/dl)")
        if clean_vitals.get("oldpeak", 0) >= 1.5:
            triggers.append(f"Ischemic ST Depression ({clean_vitals['oldpeak']:.1f} mm)")
        if clean_vitals.get("ca", 0) > 0:
            triggers.append(f"{int(clean_vitals['ca'])} obstructed major vessel(s)")
        if clean_vitals.get("exang", 0) == 1:
            triggers.append("Exercise-induced angina detected")
        if clean_vitals.get("thalach", 0) < 120:
            triggers.append(f"Low cardiac reserve ({int(clean_vitals['thalach'])} bpm)")

        recs = get_clinical_recommendations(pred, prob, clean_vitals)

        return jsonify(
            {
                "success": True,
                "prediction": pred,
                "probability": round(prob, 4),
                "probability_percent": round(prob * 100, 1),
                "risk_label": "High Risk" if pred == 1 else "Low Risk",
                "risk_class": "danger" if pred == 1 else "success",
                "triggers": triggers,
                "recommendations": recs,
                "vitals": clean_vitals,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/patients/search")
def search_patients():
    query = request.args.get("q", "").strip().lower()
    filter_type = request.args.get("filter", "all").strip().lower()

    if not os.path.exists(DATA_PATH):
        return jsonify({"results": []})

    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]
    except Exception:
        return jsonify({"results": []})

    results = []

    # Search dataset rows (PID-001 to PID-303) exclusively
    q_digits = query.replace("pid-", "").replace("pid", "").replace("#", "")
    for idx, row in df.iterrows():
        pid_code = f"PID-{idx+1:03d}"
        is_male = int(row["sex"]) == 1
        pool = MALE_NAMES if is_male else FEMALE_NAMES
        name = pool[idx % len(pool)]
        age = int(row["age"])
        is_high_risk = bool(row["oldpeak"] >= 1.5 or row["ca"] > 0 or row["trestbps"] >= 145)

        # Check filter
        if filter_type == "female" and is_male:
            continue
        if filter_type == "male" and not is_male:
            continue
        if filter_type == "elderly" and age < 60:
            continue
        if filter_type == "high" and not is_high_risk:
            continue

        matches = False
        if not query:
            if len(results) < 15:
                matches = True
        else:
            if query in pid_code.lower() or (q_digits.isdigit() and int(q_digits) == (idx + 1)):
                matches = True
            elif query in name.lower():
                matches = True
            elif query == str(age):
                matches = True
            elif (query in ["male", "m"] and is_male) or (query in ["female", "f"] and not is_male):
                matches = True
            elif query in ["high", "risk", "ischemia"] and is_high_risk:
                matches = True

        if matches:
            results.append(
                {
                    "id": pid_code,
                    "pid": pid_code,
                    "name": name,
                    "age": age,
                    "sex": "Male" if is_male else "Female",
                    "trestbps": int(row["trestbps"]),
                    "chol": int(row["chol"]),
                    "oldpeak": float(row["oldpeak"]),
                    "is_preset": False,
                    "is_high_risk": bool(is_high_risk),
                    "url": f"/dashboard?patient={pid_code}",
                }
            )
            if len(results) >= 25:
                break

    return jsonify({"results": results})


@app.route("/api/analytics-data")
def api_analytics_data():
    """Serves summary statistical data for interactive Chart.js graphs."""
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "Dataset not found"}), 404

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    # Age bins vs Disease
    bins = [20, 35, 45, 55, 65, 80]
    labels = ["20-34", "35-44", "45-54", "55-64", "65+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    age_dist = df.groupby(["age_group", "target"], observed=False).size().unstack(fill_value=0)

    # Chest pain distribution
    cp_names = {0: "Typical", 1: "Atypical", 2: "Non-Anginal", 3: "Asymptomatic"}
    df["cp_name"] = df["cp"].map(cp_names)
    cp_dist = df.groupby(["cp_name", "target"], observed=False).size().unstack(fill_value=0)

    return jsonify(
        {
            "age_groups": labels,
            "age_healthy": age_dist[0].tolist() if 0 in age_dist.columns else [],
            "age_diseased": age_dist[1].tolist() if 1 in age_dist.columns else [],
            "cp_types": list(cp_dist.index),
            "cp_healthy": cp_dist[0].tolist() if 0 in cp_dist.columns else [],
            "cp_diseased": cp_dist[1].tolist() if 1 in cp_dist.columns else [],
            "cholesterol_healthy": df[df["target"] == 0]["chol"].tolist(),
            "cholesterol_diseased": df[df["target"] == 1]["chol"].tolist(),
            "hr_healthy": df[df["target"] == 0]["thalach"].tolist(),
            "hr_diseased": df[df["target"] == 1]["thalach"].tolist(),
        }
    )


@app.route("/api/batch-process", methods=["POST"])
def api_batch_process():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        batch_df = pd.read_csv(file)
        batch_df.columns = [c.strip().lower() for c in batch_df.columns]

        missing = [c for c in feature_columns if c not in batch_df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        scaled = scaler.transform(batch_df[feature_columns])
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1]

        batch_df["predicted_risk"] = ["High Risk" if p == 1 else "Low Risk" for p in preds]
        batch_df["risk_probability_%"] = (probs * 100).round(1)

        records = batch_df.head(50).to_dict(orient="records")
        high_risk_count = int((preds == 1).sum())
        total_count = len(batch_df)

        return jsonify(
            {
                "success": True,
                "total": total_count,
                "high_risk": high_risk_count,
                "low_risk": total_count - high_risk_count,
                "avg_risk": round(float(probs.mean() * 100), 1),
                "preview": records,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download-template")
def download_template():
    sample_df = pd.DataFrame(
        [
            {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1},
            {"age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 187, "exang": 0, "oldpeak": 3.5, "slope": 0, "ca": 0, "thal": 2},
            {"age": 41, "sex": 0, "cp": 1, "trestbps": 130, "chol": 204, "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 2, "ca": 0, "thal": 2},
            {"age": 56, "sex": 1, "cp": 1, "trestbps": 120, "chol": 236, "fbs": 0, "restecg": 1, "thalach": 178, "exang": 0, "oldpeak": 0.8, "slope": 2, "ca": 0, "thal": 2},
        ]
    )
    csv_str = sample_df.to_csv(index=False)
    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=cardiosense_batch_sample.csv"},
    )


@app.route("/api/download-report-pdf", methods=["POST", "GET"])
def download_report_pdf():
    if request.method == "POST":
        data = request.get_json() or {}
        clean_vitals = {col: float(data.get(col, 0)) for col in feature_columns}
    else:
        patient_id = request.args.get("patient", "eleanor")
        clean_vitals = PRESET_PATIENTS.get(patient_id, PRESET_PATIENTS["eleanor"])["vitals"]

    pred, prob = perform_inference(clean_vitals)

    triggers = []
    if clean_vitals.get("trestbps", 0) >= 140:
        triggers.append(f"Hypertension Stage 2 ({int(clean_vitals['trestbps'])} mmHg)")
    if clean_vitals.get("chol", 0) >= 240:
        triggers.append(f"Hypercholesterolemia ({int(clean_vitals['chol'])} mg/dl)")
    if clean_vitals.get("oldpeak", 0) >= 1.5:
        triggers.append(f"Ischemic ST Depression ({clean_vitals['oldpeak']:.1f} mm)")
    if clean_vitals.get("ca", 0) > 0:
        triggers.append(f"{int(clean_vitals['ca'])} obstructed major vessel(s)")
    if clean_vitals.get("exang", 0) == 1:
        triggers.append("Exercise-induced angina detected")
    if clean_vitals.get("thalach", 0) < 120:
        triggers.append(f"Low cardiac reserve ({int(clean_vitals['thalach'])} bpm)")

    pdf_bytes = generate_clinical_pdf(clean_vitals, pred, prob, triggers)
    age = int(clean_vitals.get("age", 50))
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment;filename=prohealth_cardio_report_{age}yo.pdf"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
