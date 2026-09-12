from __future__ import annotations

from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import MODEL_DIR, MODEL_PATH, RANDOM_STATE
from .data_pipeline import PreparedData, build_preprocessor, load_clean_data, make_prepared_data


def _model_specs() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"model__C": [0.5, 1.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"model__max_depth": [4, 6]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=160,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=2,
            ),
            {"model__max_depth": [None, 10]},
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=180,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=1.6,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=2,
                random_state=RANDOM_STATE,
            ),
            {"model__max_depth": [3, 4]},
        ),
    }


def _feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = np.abs(estimator.coef_[0])
    return (
        pd.DataFrame(
            {"feature": names, "importance": values}
        )
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )


def train_and_evaluate(prepared: PreparedData) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.features,
        prepared.target,
        test_size=0.2,
        stratify=prepared.target,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fitted_models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}
    roc_curves: dict[str, dict[str, list[float]]] = {}

    for name, (estimator, grid) in _model_specs().items():
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        prepared.numeric_columns, prepared.categorical_columns
                    ),
                ),
                ("model", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        predictions = fitted.predict(X_test)
        probabilities = fitted.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        fitted_models[name] = fitted
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "best_params": search.best_params_,
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "cv_f1": float(search.best_score_),
        }
        roc_curves[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    best_name = max(metrics, key=lambda key: metrics[key]["roc_auc"])
    best_pipeline = fitted_models[best_name]
    return {
        "model": best_pipeline,
        "model_name": best_name,
        "metrics": metrics,
        "roc_curves": roc_curves,
        "feature_importance": _feature_importance(best_pipeline).to_dict("records"),
        "feature_names": best_pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
        .tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "customer_count": len(prepared.frame),
        "churn_rate": float(prepared.target.mean()),
        "feature_columns": prepared.features.columns.tolist(),
        "training_columns": {
            "numeric": prepared.numeric_columns,
            "categorical": prepared.categorical_columns,
        },
        "class_weighting": (
            "Class weighting was used for the linear/tree models and "
            "scale_pos_weight was used for XGBoost; no synthetic rows were added."
        ),
        "dataset_source": "IBM Telco Customer Churn dataset",
    }


def train_and_save() -> dict[str, Any]:
    frame = load_clean_data()
    prepared = make_prepared_data(frame)
    bundle = train_and_evaluate(prepared)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train() -> dict[str, Any]:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save()
