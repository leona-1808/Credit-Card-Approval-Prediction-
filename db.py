"""
db.py — SQLite storage for prediction history + dashboard stats.

Keeps everything in a single file, credit_predictions.db, created
automatically on first run. No external DB server needed.
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "credit_predictions.db"

# Maps the numeric codes used by predict.html's <select> options to
# human-readable labels, so the stored data (and Power BI dashboard)
# shows "Male" / "Working" / "Married" instead of raw 0/1/2 codes.
GENDER_MAP = {0: "Female", 1: "Male"}
YES_NO_MAP = {0: "No", 1: "Yes"}
INCOME_TYPE_MAP = {
    0: "Commercial Associate", 1: "Pensioner", 2: "State Servant",
    3: "Student", 4: "Working",
}
FAMILY_STATUS_MAP = {
    0: "Civil Marriage", 1: "Married", 2: "Separated",
    3: "Single / Not Married", 4: "Widow",
}
HOUSING_TYPE_MAP = {
    0: "Co-op Apartment", 1: "House / Apartment", 2: "Municipal Apartment",
    3: "Office Apartment", 4: "Rented Apartment", 5: "With Parents",
}


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _add_column_if_missing(conn, table, column, coltype):
    """Safely adds a column to an existing table without losing data.
    SQLite has no 'ADD COLUMN IF NOT EXISTS', so we check first."""
    existing = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                result TEXT NOT NULL,          -- 'Approved' or 'Rejected'
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                age INTEGER,
                income REAL,
                employment_years INTEGER,
                credit_history_months INTEGER
            )
        """)
        # Migration: add the richer demographic/application fields to
        # any existing database without dropping already-stored rows.
        _add_column_if_missing(conn, "predictions", "gender", "TEXT")
        _add_column_if_missing(conn, "predictions", "own_car", "TEXT")
        _add_column_if_missing(conn, "predictions", "own_realty", "TEXT")
        _add_column_if_missing(conn, "predictions", "children", "INTEGER")
        _add_column_if_missing(conn, "predictions", "income_type", "TEXT")
        _add_column_if_missing(conn, "predictions", "family_status", "TEXT")
        _add_column_if_missing(conn, "predictions", "housing_type", "TEXT")
        _add_column_if_missing(conn, "predictions", "work_phone", "TEXT")
        _add_column_if_missing(conn, "predictions", "phone", "TEXT")


def risk_level(confidence):
    """Simple confidence-derived risk banding."""
    if confidence >= 85:
        return "Low Risk", "🟢"
    elif confidence >= 60:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"


def insert_prediction(
    result, confidence, age, income, employment_years, credit_history_months,
    timestamp=None,
    gender=None, own_car=None, own_realty=None, children=None,
    income_type=None, family_status=None, housing_type=None,
    work_phone=None, phone=None,
):
    level, emoji = risk_level(confidence)
    ts = timestamp if timestamp is not None else datetime.now().isoformat(timespec="seconds")

    # Convert raw form codes (0/1/2...) to human-readable labels, if given as ints.
    gender_label = GENDER_MAP.get(gender, gender) if gender is not None else None
    own_car_label = YES_NO_MAP.get(own_car, own_car) if own_car is not None else None
    own_realty_label = YES_NO_MAP.get(own_realty, own_realty) if own_realty is not None else None
    income_type_label = INCOME_TYPE_MAP.get(income_type, income_type) if income_type is not None else None
    family_status_label = FAMILY_STATUS_MAP.get(family_status, family_status) if family_status is not None else None
    housing_type_label = HOUSING_TYPE_MAP.get(housing_type, housing_type) if housing_type is not None else None
    work_phone_label = YES_NO_MAP.get(work_phone, work_phone) if work_phone is not None else None
    phone_label = YES_NO_MAP.get(phone, phone) if phone is not None else None

    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (timestamp, result, confidence, risk_level, age, income,
                employment_years, credit_history_months, gender, own_car,
                own_realty, children, income_type, family_status,
                housing_type, work_phone, phone)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts, result, confidence, f"{emoji} {level}", age, income,
                employment_years, credit_history_months, gender_label,
                own_car_label, own_realty_label, children, income_type_label,
                family_status_label, housing_type_label, work_phone_label,
                phone_label,
            ),
        )
        return cur.lastrowid


def get_prediction(pred_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
        return dict(row) if row else None


def get_all_predictions(filter_result=None, search=None):
    query = "SELECT * FROM predictions"
    conditions = []
    params = []

    if filter_result and filter_result != "all":
        conditions.append("result = ?")
        params.append(filter_result)

    if search:
        conditions.append("(timestamp LIKE ? OR result LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_recent_predictions(limit=5):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        approved = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE result = 'Approved'"
        ).fetchone()["c"]
        rejected = total - approved
        avg_conf_row = conn.execute("SELECT AVG(confidence) AS a FROM predictions").fetchone()
        avg_conf = round(avg_conf_row["a"], 1) if avg_conf_row["a"] is not None else 0

    approved_pct = round((approved / total) * 100, 1) if total else 0
    rejected_pct = round((rejected / total) * 100, 1) if total else 0

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "avg_confidence": avg_conf,
        "approved_pct": approved_pct,
        "rejected_pct": rejected_pct,
    }


def get_weekly_counts():
    """Count predictions per weekday (Mon-Sun) for all stored history."""
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts = {d: 0 for d in labels}

    with get_conn() as conn:
        rows = conn.execute("SELECT timestamp FROM predictions").fetchall()

    for row in rows:
        ts = datetime.fromisoformat(row["timestamp"])
        weekday_label = labels[ts.weekday()]
        counts[weekday_label] += 1

    max_count = max(counts.values()) if counts.values() else 0
    return [{"day": d, "count": counts[d]} for d in labels], max_count
