from flask import Flask, render_template, request, Response, send_file
import pickle
import numpy as np
import io
import csv

import db
from pdf_report import build_prediction_pdf

app = Flask(__name__)

# Load Model
model = pickle.load(open("model.pkl", "rb"))

# Create the predictions table if it doesn't exist yet
db.init_db()

RISK_CLASS_MAP = {
    "Low Risk": "risk-low",
    "Medium Risk": "risk-medium",
    "High Risk": "risk-high",
}


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    stats = db.get_stats()
    recent = db.get_recent_predictions(limit=5)
    weekly, weekly_max = db.get_weekly_counts()
    return render_template(
        "home.html",
        stats=stats,
        recent=recent,
        weekly=weekly,
        weekly_max=weekly_max,
    )


# ---------------- PREDICTION PAGE ----------------
@app.route("/predictpage")
def predictpage():
    return render_template("predict.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["AGE"])
    days_birth = -(age * 365)

    employment_years = int(request.form["EMPLOYMENT_YEARS"])
    days_employed = -(employment_years * 365)

    credit_history = int(request.form["CREDIT_HISTORY_MONTHS"])
    months_balance = -credit_history

    income = float(request.form["AMT_INCOME_TOTAL"])

    features = [
        0,      # ID

        float(request.form["CODE_GENDER"]),
        float(request.form["FLAG_OWN_CAR"]),
        float(request.form["FLAG_OWN_REALTY"]),

        float(request.form["CNT_CHILDREN"]),

        income,
        float(request.form["NAME_INCOME_TYPE"]),

        1,      # NAME_EDUCATION_TYPE (not collected on form)

        float(request.form["NAME_FAMILY_STATUS"]),

        float(request.form["NAME_HOUSING_TYPE"]),

        days_birth,
        days_employed,

        1,      # FLAG_MOBIL (not collected on form)

        float(request.form["FLAG_WORK_PHONE"]),
        float(request.form["FLAG_PHONE"]),

        0,      # FLAG_EMAIL (not collected on form)

        2,      # CNT_FAM_MEMBERS (not collected on form)

        months_balance
    ]

    final_input = np.array(features).reshape(1, -1)

    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)
    confidence = round(float(np.max(probability)) * 100, 2)

    is_approved = bool(prediction[0] == 1)
    result_word = "Approved" if is_approved else "Rejected"
    prediction_text = "✅ Credit Card Approved" if is_approved else "❌ Credit Card Rejected"

    # Save to history
    pred_id = db.insert_prediction(
        result=result_word,
        confidence=confidence,
        age=age,
        income=income,
        employment_years=employment_years,
        credit_history_months=credit_history,
        gender=int(request.form["CODE_GENDER"]),
        own_car=int(request.form["FLAG_OWN_CAR"]),
        own_realty=int(request.form["FLAG_OWN_REALTY"]),
        children=int(request.form["CNT_CHILDREN"]),
        income_type=int(request.form["NAME_INCOME_TYPE"]),
        family_status=int(request.form["NAME_FAMILY_STATUS"]),
        housing_type=int(request.form["NAME_HOUSING_TYPE"]),
        work_phone=int(request.form["FLAG_WORK_PHONE"]),
        phone=int(request.form["FLAG_PHONE"]),
    )

    level, emoji = db.risk_level(confidence)

    return render_template(
        "result.html",
        prediction_text=prediction_text,
        confidence=confidence,
        is_approved=is_approved,
        risk_level=f"{emoji} {level}",
        risk_class=RISK_CLASS_MAP.get(level, "risk-medium"),
        summary={
            "age": age,
            "income": int(income),
            "credit_history_months": credit_history,
            "employment_years": employment_years,
        },
        pred_id=pred_id,
    )


# ---------------- PREDICTION HISTORY ----------------
@app.route("/history")
def history():
    current_filter = request.args.get("filter", "all")
    q = request.args.get("q", "").strip()

    predictions = db.get_all_predictions(
        filter_result=current_filter if current_filter != "all" else None,
        search=q if q else None,
    )

    return render_template(
        "history.html",
        predictions=predictions,
        current_filter=current_filter,
        q=q,
    )


@app.route("/history/csv")
def history_csv():
    current_filter = request.args.get("filter", "all")
    q = request.args.get("q", "").strip()

    predictions = db.get_all_predictions(
        filter_result=current_filter if current_filter != "all" else None,
        search=q if q else None,
    )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "ID", "Date", "Result", "Confidence", "Risk Level", "Age", "Income",
        "Employment Years", "Credit History (Months)", "Gender", "Own Car",
        "Own Realty", "Children", "Income Type", "Family Status",
        "Housing Type", "Work Phone", "Phone",
    ])
    for p in predictions:
        writer.writerow([
            p["id"], p["timestamp"], p["result"], p["confidence"], p["risk_level"],
            p["age"], p["income"], p["employment_years"], p["credit_history_months"],
            p.get("gender"), p.get("own_car"), p.get("own_realty"), p.get("children"),
            p.get("income_type"), p.get("family_status"), p.get("housing_type"),
            p.get("work_phone"), p.get("phone"),
        ])

    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
    )


# ---------------- DOWNLOAD SINGLE PREDICTION AS PDF ----------------
@app.route("/download/<int:pred_id>")
def download_pdf(pred_id):
    pred = db.get_prediction(pred_id)
    if pred is None:
        return "Prediction not found", 404

    pdf_bytes = build_prediction_pdf(pred)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"prediction_report_{pred_id}.pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
