# Retent — Customer Churn Risk Analyzer

Streamlit decision workspace that trains explainable churn models and turns customer risk into retention actions.

## Run & Operate

- `.pythonlibs/bin/streamlit run app.py --server.port 5000 --server.address 0.0.0.0` — run the dashboard
- `python train.py` — retrain the churn models and write the Joblib artifact
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- No external secrets are required. The IBM Telco dataset is bundled under `data/`.

## Stack

- Python 3.11, Streamlit, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Plotly, Joblib
- SQLite for prediction history
- pnpm workspace files remain available for the starter API and reusable libraries

## Where things live

- `app.py` — Streamlit navigation and page rendering
- `churn_analysis/data_pipeline.py` — cleaning, deterministic feature engineering, and leakage-safe preprocessing
- `churn_analysis/modeling.py` — cross-validated tuning and model selection
- `churn_analysis/explainability.py` — SHAP local/global explanations
- `churn_analysis/retention.py` — transparent rule-based recommendations
- `churn_analysis/database.py` — SQLite history persistence
- `README.md` — full project structure, methodology, and page guide

## Architecture decisions

- Class weighting was chosen over SMOTE to preserve the observed population and keep the viva explanation transparent.
- Preprocessing lives inside each scikit-learn pipeline so encoders and scalers fit only on training data.
- XGBoost 2.1.4 is pinned because it is compatible with the installed SHAP 0.46 tree explainer.
- Retention recommendations are rule-based and separate from model probability so actions are inspectable and defensible.

## Product

- Executive portfolio overview with risk distribution and contract churn
- Manual scoring and what-if analysis with SHAP explanations
- Customer segment analytics with downloadable scored rows
- Model comparison, ROC curves, confusion matrix, and feature importance
- SQLite-backed prediction history and downloadable records

## User preferences

- Build in explicit phases with modular files instead of one large script.

## Gotchas

- The first dashboard launch trains four models when `models/churn_model.joblib` does not exist.
- If the model or dataset changes, restart the `Retent Dashboard` workflow to reload Streamlit caches.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
