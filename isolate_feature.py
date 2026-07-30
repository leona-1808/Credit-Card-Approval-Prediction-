"""
isolate_feature.py

Same weak applicant as before. This time we vary ONE field at a time
(everything else held fixed) across a wide realistic range, to see
which fields actually move the prediction at all.
"""

import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

base = [
    0,      # 0  ID
    0,      # 1  CODE_GENDER
    0,      # 2  FLAG_OWN_CAR
    0,      # 3  FLAG_OWN_REALTY
    0,      # 4  CNT_CHILDREN
    20000,  # 5  AMT_INCOME_TOTAL
    4,      # 6  NAME_INCOME_TYPE
    1,      # 7  NAME_EDUCATION_TYPE
    1,      # 8  NAME_FAMILY_STATUS
    1,      # 9  NAME_HOUSING_TYPE
    -21*365,# 10 DAYS_BIRTH
    0,      # 11 DAYS_EMPLOYED
    1,      # 12 FLAG_MOBIL
    0,      # 13 FLAG_WORK_PHONE
    0,      # 14 FLAG_PHONE
    0,      # 15 FLAG_EMAIL
    2,      # 16 CNT_FAM_MEMBERS
    0,      # 17 MONTHS_BALANCE
]

def run(features):
    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]
    prob = max(model.predict_proba(x)[0])
    return ("Approved" if pred == 1 else "Rejected"), prob

base_label, base_conf = run(base)
print(f"Baseline (weak applicant): {base_label} ({base_conf*100:.1f}%)\n")

tests = {
    "AMT_INCOME_TOTAL (5)":  [(20000, "low"), (500000, "mid"), (2500000, "very high")],
    "DAYS_EMPLOYED (11)":    [(0, "0 yrs"), (-5*365, "5 yrs"), (-25*365, "25 yrs")],
    "MONTHS_BALANCE (17)":   [(0, "0 mo credit"), (-24, "24 mo"), (-60, "60 mo")],
    "NAME_HOUSING_TYPE (9)": [(1, "House/Apt"), (4, "Rented"), (5, "With parents")],
    "NAME_FAMILY_STATUS(8)": [(1, "Married"), (3, "Single"), (2, "Separated")],
    "CNT_CHILDREN (4)":      [(0, "0 kids"), (2, "2 kids"), (4, "4 kids")],
}

positions = {"AMT_INCOME_TOTAL (5)": 5, "DAYS_EMPLOYED (11)": 11, "MONTHS_BALANCE (17)": 17,
             "NAME_HOUSING_TYPE (9)": 9, "NAME_FAMILY_STATUS(8)": 8, "CNT_CHILDREN (4)": 4}

for name, variants in tests.items():
    pos = positions[name]
    print(f"--- Varying {name} ---")
    for val, desc in variants:
        f = base.copy()
        f[pos] = val
        label, conf = run(f)
        changed = " <-- CHANGED" if label != base_label else ""
        print(f"  {desc:>15}: {label} ({conf*100:.1f}%){changed}")
    print()