"""
test_sign_fix.py

Same weak applicant as before, but this time DAYS_BIRTH and
DAYS_EMPLOYED are POSITIVE, matching the .abs() transform applied
during training (cell 20 of the notebook). If this changes the
prediction, we've found and can fix the real bug.
"""

import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

def run(features, label):
    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]
    prob = max(model.predict_proba(x)[0])
    result = "Approved" if pred == 1 else "Rejected"
    print(f"{label}: {result} ({prob*100:.1f}%)")

# Original (negative days) — what app.py currently sends
negative_version = [
    0, 0, 0, 0, 0, 20000, 4, 1, 1, 1,
    -21*365,   # DAYS_BIRTH negative
    0,         # DAYS_EMPLOYED (0 either way)
    1, 0, 0, 0, 2, 0,
]
run(negative_version, "OLD (negative DAYS_BIRTH)")

# Corrected (positive days) — matching the .abs() applied in training
positive_version = [
    0, 0, 0, 0, 0, 20000, 4, 1, 1, 1,
    21*365,    # DAYS_BIRTH positive
    0,         # DAYS_EMPLOYED (0 either way)
    1, 0, 0, 0, 2, 0,
]
run(positive_version, "FIXED (positive DAYS_BIRTH)")

print()

# Now also test employment years with POSITIVE sign across a range
for years, desc in [(0, "0 yrs"), (5, "5 yrs"), (25, "25 yrs")]:
    f = [0, 0, 0, 0, 0, 20000, 4, 1, 1, 1, 21*365, years*365, 1, 0, 0, 0, 2, 0]
    run(f, f"FIXED, employment={desc}")