"""
test_new_model_distribution.py

Step 3 of the verification plan. Runs 500 synthetic applicants
(same 4-tier generator as seed_data.py) through the NEW model.pkl,
just to see the resulting approve/reject split. Does NOT write to
the database — this is a dry-run check before we touch anything
live.
"""

import random
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))


def random_applicant():
    tier = random.choices(
        ["strong", "average", "risky", "extreme"],
        weights=[35, 30, 20, 15],
    )[0]

    gender = random.choice([0, 1])
    not_employed = False

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
        if income_type == 1 or random.random() < 0.20:
            not_employed = True
            employment_years = 0
        else:
            employment_years = random.randint(0, 3)
        credit_history_months = random.randint(0, 10)
    else:
        own_car = random.choices([0, 1], weights=[80, 20])[0]
        own_realty = random.choices([0, 1], weights=[75, 25])[0]
        children = random.choices([0, 1, 2, 3, 4, 5], weights=[15, 15, 20, 25, 15, 10])[0]
        age = random.randint(21, 35)
        income = round(random.uniform(70000, 180000), 2)
        income_type = random.choices([0, 1, 2, 3, 4], weights=[10, 30, 5, 30, 25])[0]
        family_status = random.choices([0, 1, 2, 3, 4], weights=[15, 15, 25, 40, 5])[0]
        housing_type = random.choices([0, 1, 2, 3, 4, 5], weights=[3, 15, 7, 3, 32, 40])[0]
        if income_type == 1 or random.random() < 0.35:
            not_employed = True
            employment_years = 0
        else:
            employment_years = random.randint(0, 1)
        credit_history_months = random.randint(0, 3)

    work_phone = random.choice([0, 1])
    phone = random.choice([0, 1])

    return {
        "CODE_GENDER": gender, "FLAG_OWN_CAR": own_car, "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children, "AGE": age, "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": income_type, "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type, "EMPLOYMENT_YEARS": employment_years,
        "NOT_EMPLOYED": not_employed, "FLAG_WORK_PHONE": work_phone,
        "FLAG_PHONE": phone, "CREDIT_HISTORY_MONTHS": credit_history_months,
    }


def build_features(a):
    days_birth = a["AGE"] * 365  # POSITIVE, matches new training
    days_employed = 365243 if a.get("NOT_EMPLOYED") else a["EMPLOYMENT_YEARS"] * 365  # POSITIVE
    months_balance = -a["CREDIT_HISTORY_MONTHS"]

    return [
        float(a["CODE_GENDER"]), float(a["FLAG_OWN_CAR"]), float(a["FLAG_OWN_REALTY"]),
        float(a["CNT_CHILDREN"]), a["AMT_INCOME_TOTAL"], float(a["NAME_INCOME_TYPE"]),
        1,  # NAME_EDUCATION_TYPE
        float(a["NAME_FAMILY_STATUS"]), float(a["NAME_HOUSING_TYPE"]),
        days_birth, days_employed,
        1,  # FLAG_MOBIL
        float(a["FLAG_WORK_PHONE"]), float(a["FLAG_PHONE"]),
        0,  # FLAG_EMAIL
        2,  # CNT_FAM_MEMBERS
        months_balance,
    ]


N = 500
approved = 0
rejected = 0

for _ in range(N):
    a = random_applicant()
    x = np.array(build_features(a)).reshape(1, -1)
    pred = model.predict(x)[0]
    if pred == 1:
        approved += 1
    else:
        rejected += 1

print(f"Dry run (no DB writes) — {N} synthetic applicants through the NEW model:")
print(f"  Approved: {approved}")
print(f"  Rejected: {rejected}")