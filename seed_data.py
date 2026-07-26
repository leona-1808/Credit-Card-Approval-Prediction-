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
        float(a["FLAG_OWN_CAR"]),"""
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

    Uses four tiers instead of a simple strong/weak split, closer to
    a real applicant population:
      35% strong    — established income, employment, credit history
      30% average   — middling on most factors
      20% risky     — several weak factors together
      15% extremely risky — very weak across the board, including
                            a realistic 'not currently employed'
                            pattern (matches the real dataset's own
                            '365243 sentinel' behavior for pensioners/
                            unemployed applicants, which regular
                            'employment_years=0' does NOT replicate).

    The model still makes every decision itself — this only changes
    who's being evaluated.
    """
    tier = random.choices(
        ["strong", "average", "risky", "extreme"],
        weights=[35, 30, 20, 15],
    )[0]

    gender = random.choice([0, 1])
    not_employed = False  # tracks the sentinel case for build_features()

    if tier == "strong":
        own_car = random.choices([0, 1], weights=[30, 70])[0]
        own_realty = random.choices([0, 1], weights=[30, 70])[0]
        children = random.choices([0, 1, 2], weights=[55, 30, 15])[0]
        age = random.randint(28, 60)
        income = round(random.uniform(600000, 2500000), 2)
        income_type = random.choices([0, 1, 2, 4], weights=[30, 5, 20, 45])[0]
        family_status = random.choices([0, 1, 2, 3, 4], weights=[8, 60, 5, 22, 5])[0]
        housing_type = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 78, 5, 5, 4, 3])[0]
        employment_years = random.randint(5, 30)
        credit_history_months = random.randint(24, 60)

    elif tier == "average":
        own_car = random.choices([0, 1], weights=[50, 50])[0]
        own_realty = random.choices([0, 1], weights=[45, 55])[0]
        children = random.choices([0, 1, 2, 3], weights=[45, 30, 18, 7])[0]
        age = random.randint(23, 55)
        income = round(random.uniform(250000, 650000), 2)
        income_type = random.choices([0, 1, 2, 3, 4], weights=[25, 10, 15, 5, 45])[0]
        family_status = random.choices([0, 1, 2, 3, 4], weights=[10, 45, 10, 30, 5])[0]
        housing_type = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 60, 5, 5, 15, 10])[0]
        employment_years = random.randint(2, 15)
        credit_history_months = random.randint(6, 36)

    elif tier == "risky":
        own_car = random.choices([0, 1], weights=[65, 35])[0]
        own_realty = random.choices([0, 1], weights=[60, 40])[0]
        children = random.choices([0, 1, 2, 3, 4], weights=[20, 25, 25, 20, 10])[0]
        age = random.randint(21, 40)
        income = round(random.uniform(120000, 320000), 2)
        income_type = random.choices([0, 1, 2, 3, 4], weights=[15, 20, 10, 20, 35])[0]
        family_status = random.choices([0, 1, 2, 3, 4], weights=[15, 25, 20, 35, 5])[0]
        housing_type = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 25, 10, 5, 30, 25])[0]
        # ~35% chance of 'not employed' pattern (matches real Pensioner/unemployed rows)
        if income_type == 1 or random.random() < 0.20:  # Pensioner or unlucky draw
            not_employed = True
            employment_years = 0
        else:
            employment_years = random.randint(0, 3)
        credit_history_months = random.randint(0, 10)

    else:  # extreme
        own_car = random.choices([0, 1], weights=[80, 20])[0]
        own_realty = random.choices([0, 1], weights=[75, 25])[0]
        children = random.choices([0, 1, 2, 3, 4, 5], weights=[15, 15, 20, 25, 15, 10])[0]
        age = random.randint(21, 35)
        income = round(random.uniform(70000, 180000), 2)
        income_type = random.choices([0, 1, 2, 3, 4], weights=[10, 30, 5, 30, 25])[0]
        family_status = random.choices([0, 1, 2, 3, 4], weights=[15, 15, 25, 40, 5])[0]
        housing_type = random.choices([0, 1, 2, 3, 4, 5], weights=[3, 15, 7, 3, 32, 40])[0]
        # ~50% chance of 'not employed' pattern — even more common at this tier
        if income_type == 1 or random.random() < 0.35:
            not_employed = True
            employment_years = 0
        else:
            employment_years = random.randint(0, 1)
        credit_history_months = random.randint(0, 3)

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
        "NOT_EMPLOYED": not_employed,
        "FLAG_WORK_PHONE": work_phone,
        "FLAG_PHONE": phone,
        "CREDIT_HISTORY_MONTHS": credit_history_months,
    }


def build_features(a):
    """Same feature order app.py uses for real form submissions."""
    days_birth = -(a["AGE"] * 365)

    # Matches the real dataset's own convention: unemployed/pensioner
    # applicants get the special 365243 sentinel value instead of a
    # small number of days. Confirmed directly from real held-out
    # data earlier (a real applicant showed DAYS_EMPLOYED=365243,
    # POSITIVE) — using that exact confirmed value here, not a guess.
    if a.get("NOT_EMPLOYED"):
        days_employed = 365243
    else:
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