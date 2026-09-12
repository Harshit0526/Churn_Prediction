# Retent — Intelligent Customer Churn Risk Analysis

Retent is a Streamlit decision workspace for customer churn analysis and retention planning. It trains real classification models on IBM's public Telco Customer Churn dataset, explains individual scores with SHAP, and translates risk into transparent rule-based actions.

## Project structure

```text
.
├── app.py                         # Streamlit pages and interaction layer
├── train.py                       # Reproducible training entry point
├── churn_analysis/
│   ├── config.py                  # Paths, thresholds, and shared settings
│   ├── data_pipeline.py           # Download, cleaning, feature engineering, preprocessing
│   ├── modeling.py                # CV tuning, evaluation, and Joblib artifact creation
│   ├── explainability.py          # Local/global SHAP analysis
│   ├── retention.py               # Transparent, non-ML retention rules
│   └── database.py                # SQLite prediction history
├── data/
│   └── Telco-Customer-Churn.csv   # IBM Telco dataset
├── models/                        # Generated trained-model artifacts
└── churn_history.db               # Created on first saved prediction
```

## Run

```bash
streamlit run app.py --server.port 5000
```

The first launch trains four models if `models/churn_model.joblib` is not present. To retrain:

```bash
python train.py
```

## Modeling decisions

- `customerID` and the outcome column are excluded from training features.
- `TotalCharges` is parsed as numeric and missing values are imputed inside the preprocessing pipeline.
- Deterministic features include tenure bucket, average monthly charge, service count, and household support.
- Encoders and scalers are fit only on the training split because they live inside each model's `Pipeline`.
- Class weighting was chosen instead of SMOTE. The original row distribution stays intact, the approach is easier to defend in a viva, and the decision boundary still penalizes missed churn cases.
- Logistic Regression, Decision Tree, Random Forest, and XGBoost are tuned with stratified cross-validation and compared on a held-out test set.
- The selected model is saved with Joblib. SHAP is the primary explanation method; built-in feature importance is shown as a secondary comparison.

## Dashboard pages

1. **Executive Dashboard** — portfolio KPIs, risk distribution, contract churn, and high-risk customers.
2. **Churn Prediction** — score a customer, show local SHAP drivers, generate a retention plan, and log the case.
3. **What-If Analysis** — change customer attributes and see the live probability update.
4. **Customer Analytics** — filter segments, inspect interactive Plotly views, and download scored rows.
5. **Model Performance** — compare metrics, ROC curves, confusion matrix, and feature importance.
6. **Prediction History** — review and download SQLite-backed manual prediction records.

## Risk thresholds

- Low: probability below 35%
- Medium: 35% through 64.9%
- High: 65% or above

These thresholds are intentionally explicit and can be changed in `churn_analysis/config.py` without retraining the model.