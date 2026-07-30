"""
confirm_id_leakage.py

Tests your existing model.pkl with the SAME weak applicant,
changing ONLY the ID value. If the prediction/confidence shifts
meaningfully just from changing ID, that confirms the model is
keying off ID instead of the applicant's real characteristics.
"""

import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

# The same extremely weak applicant you tested manually:
# ₹20,000 income, 0 years employment, 0 months credit history
base_features = [
    0,      # ID  <-- this is the only thing we'll change
    0,      # CODE_GENDER
    0,      # FLAG_OWN_CAR
    0,      # FLAG_OWN_REALTY
    0,      # CNT_CHILDREN
    20000,  # AMT_INCOME_TOTAL
    4,      # NAME_INCOME_TYPE (Working)
    1,      # NAME_EDUCATION_TYPE
    1,      # NAME_FAMILY_STATUS
    1,      # NAME_HOUSING_TYPE
    -21*365,  # DAYS_BIRTH (~21 years old)
    0,      # DAYS_EMPLOYED (0 years)
    1,      # FLAG_MOBIL
    0,      # FLAG_WORK_PHONE
    0,      # FLAG_PHONE
    0,      # FLAG_EMAIL
    2,      # CNT_FAM_MEMBERS
    0,      # MONTHS_BALANCE (0 months credit history)
]

test_ids = [0, 5000, 50000, 200000, 400000]

print("Same weak applicant, only ID changes:\n")
for test_id in test_ids:
    features = base_features.copy()
    features[0] = test_id
    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0]
    label = "Approved" if pred == 1 else "Rejected"
    print(f"  ID={test_id:>7}  ->  {label}   (confidence: {max(prob)*100:.1f}%)")