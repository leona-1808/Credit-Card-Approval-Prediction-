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
    """Generates one plausible (not real) applicant profile.

    ~70% are 'typical' applicants (original balanced distribution).
    ~30% are deliberately weaker profiles — low income, little/no
    employment history, thin credit history, more dependents — so the
    model sees a realistic mix of strong and risky applicants instead
    of everyone looking similarly qualified. The model still makes
    every decision itself; this only changes who's being evaluated.
    """
    is_weak_profile = random.random() < 0.30

    gender = random.choice([0, 1])
    own_car = random.choices([0, 1], weights=[70, 30] if is_weak_profile else [40, 60])[0]
    own_realty = random.choices([0, 1], weights=[65, 35] if is_weak_profile else [35, 65])[0]

    if is_weak_profile:
        children = random.choices([0, 1, 2, 3, 4], weights=[15, 20, 25, 25, 15])[0]
        age = random.randint(21, 35)  # younger, less established
        income = round(random.uniform(80000, 220000), 2)  # low income band
        income_type = random.choices([0, 1, 2, 3, 4], weights=[15, 5, 10, 25, 45])[0]  # more students
        family_status = random.choices([0, 1, 2, 3, 4], weights=[15, 25, 20, 35, 5])[0]  # more separated/single
        housing_type = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 25, 10, 5, 30, 25])[0]  # more rented/with parents
        employment_years = random.randint(0, 2)  # little to no work history
        credit_history_months = random.randint(0, 6)  # thin credit file
    else:
        children = random.choices([0, 1, 2, 3], weights=[60, 25, 10, 5])[0]
        age = random.randint(21, 65)
        income = round(random.choice([
            random.uniform(300000, 600000),   # mid income band
            random.uniform(600000, 2500000),  # high income band
        ]), 2)
        income_type = random.choices([0, 1, 2, 3, 4], weights=[25, 10, 15, 2, 48])[0]
        family_status = random.choices([0, 1, 2, 3, 4], weights=[10, 50, 8, 27, 5])[0]
        housing_type = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 70, 5, 5, 8, 7])[0]
        employment_years = random.randint(2, 30)
        credit_history_months = random.randint(6, 60)

    work_phone = random.choice([0, 1])
    phone = random.choice([0, 1])

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
        gender=applicant["CODE_GENDER"],
        own_car=applicant["FLAG_OWN_CAR"],
        own_realty=applicant["FLAG_OWN_REALTY"],
        children=applicant["CNT_CHILDREN"],
        income_type=applicant["NAME_INCOME_TYPE"],
        family_status=applicant["NAME_FAMILY_STATUS"],
        housing_type=applicant["NAME_HOUSING_TYPE"],
        work_phone=applicant["FLAG_WORK_PHONE"],
        phone=applicant["FLAG_PHONE"],
    )

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{N} done")

print(f"\nDone. {approved_count} approved, {rejected_count} rejected.")
print("Refresh your home page / history page to see the new data.")