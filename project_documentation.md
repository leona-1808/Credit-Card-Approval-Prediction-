# Credit Card Approval Prediction System — Full Project Documentation

## 1. Project Overview

An end-to-end machine learning project that predicts whether a credit card application will be **approved** or **rejected**, based on applicant demographics, employment, and credit history. The project has two halves:

1. **Model development** (Jupyter notebook) — data merging, cleaning, encoding, handling class imbalance, training and comparing four classification models.
2. **Web application** (Flask) — a form-based UI where a user enters applicant details and gets a live prediction, plus a dashboard showing model performance and a history of past predictions.

---

## 2. Dataset & Data Pipeline

**Source data:** two Kaggle credit card approval tables merged on applicant `ID`:
- `application_df` — applicant demographic/financial data (gender, income, family status, housing, etc.)
- `credit_df` — monthly credit account status history

**Merge:**
```python
final_df = application_df.merge(credit_df, how='left', on='ID')
```

**Target variable — `STATUS_BIN`:** a binarized version of the credit account status column, derived from `STATUS`, representing whether the applicant is a "good" (approve) or "bad" (reject) credit risk. Rows with no `STATUS_BIN` (no credit history match) were dropped:
```python
final_df = final_df.dropna(subset=['STATUS_BIN'])
```

**Final dataset size:** 777,715 merged records after cleaning.

**Categorical encoding:** all object-dtype columns were label-encoded:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_cols = final_df.select_dtypes(include='object').columns
for col in categorical_cols:
    final_df[col] = le.fit_transform(final_df[col].astype(str))
```

**Final feature set (18 features fed to the model):**
```
ID, CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN,
AMT_INCOME_TOTAL, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE,
NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, DAYS_BIRTH, DAYS_EMPLOYED,
FLAG_MOBIL, FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL,
CNT_FAM_MEMBERS, MONTHS_BALANCE
```
(`ID` is included as a raw column in the training matrix but carries no predictive signal — it's passed through as a placeholder `0` in the live app.)

**Train/test split:** 80/20, stratified on the target to preserve class balance:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Class imbalance:** the dataset is heavily imbalanced — roughly 153,000 majority-class rows vs. ~2,300 minority-class rows in the test split alone. This imbalance is the central challenge of the whole project and drives every modeling decision below.

**A real bug found and fixed during development:** an early train/test split only dropped `STATUS_BIN` from the feature set but *not* `STATUS` itself, leaving a raw string column (values like `'C'`, `'X'`, `'0'`–`'5'`) in `X_train`. This caused `SMOTE.fit_resample()` to throw `ValueError: could not convert string to float: 'C'`, since SMOTE requires fully numeric input. Fix: drop both `STATUS` and `STATUS_BIN` before splitting.

---

## 3. Models Trained & Compared

Four classifiers were trained and evaluated on the same test set:

| Model | Accuracy | Recall (minority class) | Verdict |
|---|---|---|---|
| **Logistic Regression** | 98.53% | **0%** | Misleading — predicted every single row as the majority class. Never once identified a minority-class (high-risk) applicant. Accuracy is meaningless here. |
| **Decision Tree** | 98.28% | 43% | Reasonable minority detection, but a single tree tends to overfit. |
| **Random Forest** (no rebalancing) | **98.68%** (highest raw accuracy) | 39% | Best headline accuracy, but still misses over 60% of true minority-class cases. |
| **Random Forest + SMOTE** | 96.48% | **56%** ✅ Final model | Lower raw accuracy, but the largest jump in minority-class recall — the model actually selected for production. |

**Why accuracy alone was rejected as the selection metric:** with a ~67:1 class imbalance, a model can hit 98%+ accuracy just by always predicting the majority class (this is literally what Logistic Regression did). Accuracy in this context rewards ignoring the minority class entirely, which is the opposite of what a credit-risk system needs to do. Recall on the minority class — how many true high-risk applicants the model actually catches — is the metric that matters for this problem.

**SMOTE (Synthetic Minority Over-sampling Technique):** applied only to the training set (never the test set, to avoid leakage), it synthetically generates new minority-class examples by interpolating between existing ones, giving the Random Forest a balanced training distribution to learn from:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, y_train_smote)
```

**Final model selected:** Random Forest trained on the SMOTE-balanced data — traded ~2.2 points of raw accuracy for a 17-point gain in minority-class recall (39% → 56%), which is the right trade for a risk-detection use case where missing a bad applicant is more costly than a lower overall score.

**Model persisted with:**
```python
import pickle
pickle.dump(rf, open("model.pkl", "wb"))
```
*(Worth noting for your own diligence: the notebook has several `pickle.dump` calls at different points in development. Before treating any deployment as final, it's worth re-confirming `model.pkl` was saved from the SMOTE-trained Random Forest specifically, e.g. by checking `model.n_estimators` and testing a known prediction.)*

---

## 4. Web Application — Architecture

Built with **Flask**, using SQLite for lightweight, zero-setup persistence of prediction history and live dashboard stats.

### File structure
```
project/
├── app.py                  # Flask routes — the application entrypoint
├── db.py                   # SQLite helper layer (all DB reads/writes)
├── pdf_report.py            # Generates a PDF report for a single prediction
├── model.pkl                # Trained Random Forest + SMOTE model
├── requirements.txt          # Flask, numpy, scikit-learn, fpdf2
├── credit_predictions.db     # Auto-created SQLite DB (prediction history)
├── templates/
│   ├── base.html            # Shared layout: navbar, dark mode toggle, block content
│   ├── home.html             # Dashboard: model stats, live stats, charts, recent predictions
│   ├── predict.html          # Input form (personal/employment/property/contact/credit sections)
│   ├── result.html           # Prediction result: verdict, confidence, risk badge, summary, downloads
│   └── history.html          # Full prediction history: filter, search, CSV export
└── static/
    ├── css/style.css         # Full design system (light + dark themes)
    └── js/theme.js            # Dark mode toggle logic (localStorage persistence)
```

### `app.py` — routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Renders `home.html` with live dashboard stats (`db.get_stats()`), recent predictions (`db.get_recent_predictions()`), and weekly activity chart data (`db.get_weekly_counts()`). |
| `/predictpage` | GET | Renders the input form (`predict.html`). |
| `/predict` | POST | Reads all form fields, builds the 18-feature vector in the exact order the model was trained on, calls `model.predict()` / `model.predict_proba()`, computes a confidence score and risk band, **saves the prediction to SQLite**, and renders `result.html`. |
| `/history` | GET | Queries `db.get_all_predictions()` with optional `filter` (Approved/Rejected/all) and `q` (search) query params; renders `history.html`. |
| `/history/csv` | GET | Streams the (optionally filtered) prediction history as a downloadable CSV. |
| `/download/<id>` | GET | Generates and streams a one-page PDF report for a single stored prediction, via `pdf_report.build_prediction_pdf()`. |

**Feature vector construction (in `/predict`):** age is converted to `DAYS_BIRTH` (negative days, matching the training data's convention), employment years to `DAYS_EMPLOYED`, and credit history months to `MONTHS_BALANCE`. Fields not collected on the form (`NAME_EDUCATION_TYPE`, `FLAG_MOBIL`, `FLAG_EMAIL`, `CNT_FAM_MEMBERS`) are passed as fixed defaults, matching the exact column order the model expects.

*(A real bug caught and fixed during development: the form collected `CNT_CHILDREN`, `NAME_HOUSING_TYPE`, `FLAG_WORK_PHONE`, and `FLAG_PHONE`, but an earlier version of `app.py` hardcoded these to defaults instead of reading them from the submitted form — meaning user input for those four fields was silently ignored. Fixed to read all four from `request.form`.)*

### `db.py` — persistence layer

SQLite table `predictions`:
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    result TEXT NOT NULL,           -- 'Approved' or 'Rejected'
    confidence REAL NOT NULL,
    risk_level TEXT NOT NULL,       -- e.g. '🟢 Low Risk'
    age INTEGER,
    income REAL,
    employment_years INTEGER,
    credit_history_months INTEGER
)
```

Key functions:
- `insert_prediction(...)` — saves a new row, computing risk level from confidence
- `risk_level(confidence)` — bands confidence into Low (≥85%) / Medium (60–84%) / High (<60%) risk
- `get_stats()` — total / approved / rejected counts + average confidence, powering the home page's Live Statistics cards
- `get_weekly_counts()` — buckets all predictions by weekday, powering the "Predictions This Week" bar chart
- `get_all_predictions(filter_result, search)` — powers the History page's filter/search
- `get_recent_predictions(limit)` — last N predictions for the home page's Recent Predictions list

### `pdf_report.py`

Uses `fpdf2` to generate a single-page PDF summary of one prediction: verdict banner (color-coded), confidence score, risk level, and an application summary table (age, income, employment, credit history). Built to only use Latin-1-safe characters (no em dashes or emoji) since the core PDF fonts can't render them — this was caught and fixed during testing.

### Frontend / design system

- **Color palette:** navy (`#1B2A5B` / `#0F1A3D`) as the primary brand color, gold/amber as an accent for the "final model" / approved states, sage green and coral/red for positive and negative signals respectively (recall bars, risk badges, approve/reject banners).
- **Typography:** `Fraunces` (serif) for headings, `IBM Plex Sans` for body text, `IBM Plex Mono` for all numeric/data values — giving a visual distinction between narrative content and data.
- **Dark mode:** toggled via a navbar button, persisted with `localStorage`, implemented through a `data-theme="dark"` attribute on `<html>` with CSS variable overrides. (This went through a couple of rounds of contrast fixes — several elements were pinned to a fixed dark color that became invisible against the dark theme's near-black background; fixed with theme-specific overrides on headings, form section titles, stat card numbers, and step labels.)
- **Home page sections:** Project Overview (static model/dataset stats) → Live Statistics (real counts from the DB) → Approved/Rejected + weekly activity charts (pure CSS bar charts, no chart library) → Model Comparison table (all four models, with the winning row highlighted) → Recent Predictions → How It Works (4-step flow) → CTA to the prediction form.

---

## 5. Known Limitations / Things Worth Mentioning If Asked

- **SQLite is intentionally a demo-scale choice.** It requires zero setup and pairs well with a single-instance deployment, but doesn't handle concurrent writers well — deliberately deferred; a future "dashboard" phase would move to PostgreSQL.
- **Accuracy vs. recall trade-off is the core modeling story** — worth being able to explain clearly: raw accuracy is a poor metric under severe class imbalance, and the project explicitly chose the model with better minority-class recall over the model with the highest accuracy.
- **`FLAG_MOBIL`, `NAME_EDUCATION_TYPE`, `FLAG_EMAIL`, `CNT_FAM_MEMBERS`** are not collected from the user and are passed as fixed defaults — a reasonable simplification for a demo, but a genuine production system would need to either collect these or justify omitting them.
- **Deployment target:** PythonAnywhere (free tier), chosen specifically because its persistent disk keeps the SQLite prediction history intact across restarts — unlike Render/Railway's free tiers, which reset the filesystem on redeploy.

---

## 6. Tech Stack Summary

- **Data/ML:** pandas, scikit-learn (LabelEncoder, train_test_split, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, classification metrics), imbalanced-learn (SMOTE)
- **Backend:** Flask, SQLite (via Python's built-in `sqlite3`), fpdf2 (PDF generation)
- **Frontend:** Jinja2 templates, vanilla CSS (custom design system, no framework), vanilla JS (dark mode toggle only)
- **Deployment:** PythonAnywhere (Python WSGI hosting with persistent storage)
