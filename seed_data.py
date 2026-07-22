"""
seed_data.py

Generates realistic fake applicants, runs them through your REAL
trained model (model.pkl), and saves the results into your database
— exactly like someone using the web form, just automated.

Run this in the same folder as app.py, db.py, and model.pkl.

Usage:
    python seed_data.py 100        <- creates 100 predictions
    python seed_data.py            <- defaults to 50
"""

import sys
import pickle
import random
import numpy as np
from datetime import datetime, timedelta

import db

# ---- how many to generate ----
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50

# ---- load your real trained model ----
model = pickle.load(open("model.pkl", "rb"))

db.init_db()


def random_applicant():
    """Generates one plausible (not real) applicant profile."""
    gender = random.choice([0, 1])
    own_car = random.choice([0, 1])
    own_realty = random.choice([0, 1])
    children = random.choices([0, 1, 2, 3], weights=[60, 25, 10, 5])[0]
    age = random.randint(21, 65)
    income = round(random.choice([
        random.uniform(150000, 400000),   # lower income band
        random.uniform(400000, 900000),   # mid income band
        random.uniform(900000, 2500000),  # high income band
    ]), 2)
    income_type = random.choices([0, 1, 2, 3, 4], weights=[25, 10, 15, 5, 45])[0]  # weighted toward "Working"
    family_status = random.choices([0, 1, 2, 3, 4], weights=[10, 45, 10, 30, 5])[0]
    housing_type = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 65, 5, 5, 10, 10])[0]
    employment_years = random.randint(0, 30)
    work_phone = random.choice([0, 1])
    phone = random.choice([0, 1])
    credit_history_months = random.randint(0, 60)

    return {
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children,
        "AGE": age,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": income_type,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "EMPLOYMENT_YEARS": employment_years,
        "FLAG_WORK_PHONE": work_phone,
        "FLAG_PHONE": phone,
        "CREDIT_HISTORY_MONTHS": credit_history_months,
    }


def build_features(a):
    """Same feature order app.py uses for real form submissions."""
    days_birth = -(a["AGE"] * 365)
    days_employed = -(a["EMPLOYMENT_YEARS"] * 365)
    months_balance = -a["CREDIT_HISTORY_MONTHS"]

    return [
        0,  # ID
        float(a["CODE_GENDER"]),
        float(a["FLAG_OWN_CAR"]),
        float(a["FLAG_OWN_REALTY"]),
        float(a["CNT_CHILDREN"]),
        a["AMT_INCOME_TOTAL"],
        float(a["NAME_INCOME_TYPE"]),
        1,  # NAME_EDUCATION_TYPE (not collected)
        float(a["NAME_FAMILY_STATUS"]),
        float(a["NAME_HOUSING_TYPE"]),
        days_birth,
        days_employed,
        1,  # FLAG_MOBIL (not collected)
        float(a["FLAG_WORK_PHONE"]),
        float(a["FLAG_PHONE"]),
        0,  # FLAG_EMAIL (not collected)
        2,  # CNT_FAM_MEMBERS (not collected)
        months_balance,
    ]


print(f"Generating {N} predictions using your real model...")

approved_count = 0
rejected_count = 0

for i in range(N):
    applicant = random_applicant()
    features = build_features(applicant)
    final_input = np.array(features).reshape(1, -1)

    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)
    confidence = round(float(np.max(probability)) * 100, 2)

    is_approved = bool(prediction[0] == 1)
    result_word = "Approved" if is_approved else "Rejected"
    approved_count += is_approved
    rejected_count += not is_approved

    # spread timestamps across the last 14 days, random time of day,
    # so the "Predictions This Week" chart looks like real usage
    days_ago = random.randint(0, 13)
    random_time = datetime.now() - timedelta(
        days=days_ago,
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )

    db.insert_prediction(
        result=result_word,
        confidence=confidence,
        age=applicant["AGE"],
        income=applicant["AMT_INCOME_TOTAL"],
        employment_years=applicant["EMPLOYMENT_YEARS"],
        credit_history_months=applicant["CREDIT_HISTORY_MONTHS"],
        timestamp=random_time.isoformat(timespec="seconds"),
    )

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{N} done")

print(f"\nDone. {approved_count} approved, {rejected_count} rejected.")
print("Refresh your home page / history page to see the new data.")
