"""
test_new_model_manual.py

Step 2 of the verification plan. Tests strong and weak applicants
against the NEW model.pkl (17 features, no ID, POSITIVE DAYS_BIRTH/
DAYS_EMPLOYED matching training). This is a standalone check —
doesn't touch app.py or your live site.
"""

import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

# New feature order (17 features, no ID):
# CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN,
# AMT_INCOME_TOTAL, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE,
# NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, DAYS_BIRTH, DAYS_EMPLOYED,
# FLAG_MOBIL, FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL,
# CNT_FAM_MEMBERS, MONTHS_BALANCE

def build(gender, own_car, own_realty, children, income, income_type,
          family_status, housing_type, age, employment_years,
          work_phone, phone, credit_history_months, not_employed=False):
    days_birth = age * 365          # POSITIVE now
    days_employed = 365243 if not_employed else employment_years * 365  # POSITIVE
    months_balance = -credit_history_months  # unchanged, matches real data

    return [
        gender, own_car, own_realty, children, income, income_type,
        1,  # NAME_EDUCATION_TYPE
        family_status, housing_type, days_birth, days_employed,
        1,  # FLAG_MOBIL
        work_phone, phone,
        0,  # FLAG_EMAIL
        2,  # CNT_FAM_MEMBERS
        months_balance,
    ]

def test(label, features):
    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]
    prob = max(model.predict_proba(x)[0])
    result = "Approved" if pred == 1 else "Rejected"
    print(f"{label}: {result} ({prob*100:.1f}%)")

print("=== 10 STRONG applicants ===")
for i in range(10):
    f = build(
        gender=i % 2, own_car=1, own_realty=1, children=0,
        income=800000 + i * 100000, income_type=4, family_status=1,
        housing_type=1, age=35 + i, employment_years=8 + i,
        work_phone=1, phone=1, credit_history_months=36,
    )
    test(f"Strong #{i+1}", f)

print("\n=== 10 WEAK applicants ===")
for i in range(10):
    f = build(
        gender=i % 2, own_car=0, own_realty=0, children=2,
        income=70000 + i * 5000, income_type=3, family_status=2,
        housing_type=4, age=22 + i, employment_years=0,
        work_phone=0, phone=0, credit_history_months=1,
        not_employed=(i % 2 == 0),
    )
    test(f"Weak #{i+1}", f)