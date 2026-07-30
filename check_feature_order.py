"""
check_feature_order.py

Run this in the same folder as model.pkl. This settles the question
directly — no notebook rerun needed.
"""

import pickle

model = pickle.load(open("model.pkl", "rb"))

print("n_features expected:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("\n✅ The model remembers its training column order:\n")
    for i, name in enumerate(model.feature_names_in_):
        print(f"  {i}: {name}")
else:
    print("\n⚠️ Model was trained on a plain array (no column names stored).")
    print("   Can't directly compare — see the manual test below instead.")

# Feature importances tell us which POSITIONS the model actually relies
# on, regardless of what name we think belongs there.
print("\nFeature importance by position (higher = more influence):")
for i, imp in enumerate(model.feature_importances_):
    print(f"  position {i}: {imp:.4f}")