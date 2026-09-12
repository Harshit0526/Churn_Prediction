from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def _pretty_name(name: str) -> str:
    cleaned = name.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("_", " ")
    return cleaned.replace("=", ": ").title()


def _explain(estimator: Any, transformed: Any) -> Any:
    """Choose a SHAP explainer compatible with the fitted estimator."""
    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
        return explainer(transformed)
    explainer = shap.Explainer(
        lambda values: estimator.predict_proba(values)[:, 1],
        transformed,
    )
    return explainer(transformed)


def local_shap_values(
    pipeline: Any, customer_frame: pd.DataFrame, max_features: int = 8
) -> pd.DataFrame:
    """Return the strongest local SHAP drivers for one prediction."""
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(customer_frame)
    feature_names = preprocessor.get_feature_names_out()

    explanation = _explain(estimator, transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    values = values[0]
    base_values = np.asarray(explanation.base_values)
    base_value = float(base_values.reshape(-1)[-1]) if base_values.size else 0.0

    result = pd.DataFrame(
        {
            "feature": [_pretty_name(name) for name in feature_names],
            "feature_key": feature_names,
            "impact": values,
            "absolute_impact": np.abs(values),
        }
    ).sort_values("absolute_impact", ascending=False)
    result["direction"] = np.where(
        result["impact"] >= 0, "Pushes toward churn", "Protects from churn"
    )
    result["base_value"] = base_value
    return result.head(max_features).reset_index(drop=True)


def global_shap_importance(
    pipeline: Any, feature_frame: pd.DataFrame, sample_size: int = 500
) -> pd.DataFrame:
    """Compute a global mean absolute SHAP importance table."""
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    sample = feature_frame.sample(
        min(sample_size, len(feature_frame)), random_state=42
    )
    transformed = preprocessor.transform(sample)
    explanation = _explain(estimator, transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    importance = np.abs(values).mean(axis=0)
    names = [_pretty_name(name) for name in preprocessor.get_feature_names_out()]
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
