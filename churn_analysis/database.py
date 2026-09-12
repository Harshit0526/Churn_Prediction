from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                customer_id TEXT,
                inputs_json TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def log_prediction(
    customer: dict[str, Any],
    probability: float,
    level: str,
    recommendations: list[dict[str, str]],
    path: Path = DB_PATH,
) -> None:
    init_db(path)
    customer_id = customer.get("customerID") or "manual-prediction"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_history
              (timestamp, customer_id, inputs_json, churn_probability, risk_level, recommendation_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                customer_id,
                json.dumps(customer, default=str),
                float(probability),
                level,
                json.dumps(recommendations),
            ),
        )
        connection.commit()


def read_history(path: Path = DB_PATH) -> pd.DataFrame:
    init_db(path)
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query(
            """
            SELECT id, timestamp, customer_id, churn_probability, risk_level,
                   recommendation_json
            FROM prediction_history
            ORDER BY timestamp DESC
            """,
            connection,
        )
