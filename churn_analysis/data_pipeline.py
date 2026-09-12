from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DATA_PATH, DATA_URL, RANDOM_STATE


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


@dataclass
class PreparedData:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]


def ensure_dataset(path: Path = DATA_PATH) -> Path:
    """Download the public IBM dataset only when it is not already present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(DATA_URL, path)
    return path


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, leakage-safe features to a cleaned customer frame."""
    result = frame.copy()
    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    result["MonthlyCharges"] = pd.to_numeric(result["MonthlyCharges"], errors="coerce")
    result["tenure_bucket"] = pd.cut(
        result["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"],
    ).astype("object")
    result["avg_monthly_charge"] = result["TotalCharges"] / result["tenure"].clip(lower=1)
    result["service_count"] = result[SERVICE_COLUMNS].apply(
        lambda row: sum(value == "Yes" for value in row), axis=1
    )
    result["household_support"] = np.where(
        (result["Partner"] == "Yes") | (result["Dependents"] == "Yes"),
        "Supported household",
        "Single customer",
    )
    return result


def load_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load, validate, clean, and feature-engineer the IBM Telco dataset."""
    csv_path = ensure_dataset(path)
    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame.drop_duplicates().copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    numeric_columns = frame.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        frame[column] = frame[column].fillna(frame[column].median())
    categorical_columns = frame.select_dtypes(include=["object"]).columns
    for column in categorical_columns:
        frame[column] = frame[column].fillna("Unknown").astype(str).str.strip()

    frame["Churn"] = frame["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return engineer_features(frame)


def make_prepared_data(frame: pd.DataFrame) -> PreparedData:
    """Separate target and identifiers after all deterministic transformations."""
    features = frame.drop(columns=["Churn", "customerID"], errors="ignore")
    target = frame["Churn"].copy()
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    return PreparedData(
        frame=frame,
        features=features,
        target=target,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """Create a preprocessor that is fit inside each model pipeline."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
