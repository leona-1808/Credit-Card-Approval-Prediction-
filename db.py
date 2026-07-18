"""
db.py — SQLite storage for prediction history + dashboard stats.

Keeps everything in a single file, credit_predictions.db, created
automatically on first run. No external DB server needed.
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "credit_predictions.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


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


def risk_level(confidence):
    """Simple confidence-derived risk banding."""
    if confidence >= 85:
        return "Low Risk", "🟢"
    elif confidence >= 60:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"


def insert_prediction(result, confidence, age, income, employment_years, credit_history_months):
    level, emoji = risk_level(confidence)
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (timestamp, result, confidence, risk_level, age, income, employment_years, credit_history_months)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                result,
                confidence,
                f"{emoji} {level}",
                age,
                income,
                employment_years,
                credit_history_months,
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