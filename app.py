from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churn_analysis.config import PREDICTION_INPUT_COLUMNS
from churn_analysis.data_pipeline import engineer_features, load_clean_data
from churn_analysis.database import log_prediction, read_history
from churn_analysis.explainability import global_shap_importance, local_shap_values
from churn_analysis.modeling import load_or_train
from churn_analysis.retention import recommend_retention, risk_color, risk_level


st.set_page_config(
    page_title="Retent | Churn Risk Intelligence",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict[str, Any]:
    return load_or_train()


@st.cache_data(show_spinner=False)
def get_customers() -> pd.DataFrame:
    return load_clean_data()


@st.cache_data(show_spinner=False)
def get_portfolio_probabilities(_model: Any, customers: pd.DataFrame) -> list[float]:
    return _model.predict_proba(customers.drop(columns=["Churn", "customerID"]))[:, 1].tolist()


def _display_name(value: str) -> str:
    return value.replace("_", " ").replace("SeniorCitizen", "Senior citizen").title()


def _customer_row(values: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame([values])
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce")
    return engineer_features(frame).drop(columns=["customerID", "Churn"], errors="ignore")


def _predict(bundle: dict[str, Any], values: dict[str, Any]) -> tuple[float, str, list[dict[str, str]]]:
    row = _customer_row(values)
    probability = float(bundle["model"].predict_proba(row)[:, 1][0])
    level = risk_level(probability)
    return probability, level, recommend_retention(values, level)


def _metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def render_prediction_result(
    bundle: dict[str, Any], values: dict[str, Any], log_result: bool = False
) -> None:
    probability, level, recommendations = _predict(bundle, values)
    row = _customer_row(values)
    st.divider()
    left, middle, right = st.columns([1.15, 1, 1.5])
    with left:
        _metric_card("Churn probability", f"{probability:.1%}")
        st.markdown(
            f"### :{risk_color(level)}[{level} risk]",
            help="Low < 35%, Medium 35–65%, High ≥ 65%.",
        )
    with middle:
        st.metric("Customer tenure", f"{values['tenure']} months")
        st.metric("Monthly charges", f"${values['MonthlyCharges']:,.2f}")
    with right:
        st.subheader("Recommended retention plan")
        for item in recommendations:
            st.write(f"**{item['priority']}** · {item['action']}")
            st.caption(item["reason"])

    try:
        drivers = local_shap_values(bundle["model"], row)
        drivers["impact_label"] = drivers["impact"].map(lambda value: f"{value:+.3f}")
        fig = px.bar(
            drivers.sort_values("impact"),
            x="impact",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Pushes toward churn": "#d95f59",
                "Protects from churn": "#2f8f83",
            },
            hover_data={"impact": ":.3f", "direction": True},
            title="Why this prediction moved",
        )
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Primary explanation: local SHAP values. Positive values push toward churn.")
    except Exception as exc:
        st.warning(f"SHAP explanation is temporarily unavailable: {exc}")

    if log_result:
        log_prediction(values, probability, level, recommendations)
        st.success("Prediction saved to history.")


def customer_form(key_prefix: str, compact: bool = False) -> dict[str, Any]:
    st.caption("Change any attribute to see how the model responds.")
    columns = st.columns(3 if not compact else 2)
    with columns[0]:
        gender = st.selectbox("Gender", ["Female", "Male"], key=f"{key_prefix}_gender")
        senior = st.selectbox(
            "Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No",
            key=f"{key_prefix}_senior",
        )
        partner = st.selectbox("Partner", ["No", "Yes"], key=f"{key_prefix}_partner")
        dependents = st.selectbox(
            "Dependents", ["No", "Yes"], key=f"{key_prefix}_dependents"
        )
        tenure = st.slider("Tenure (months)", 0, 72, 12, key=f"{key_prefix}_tenure")
        phone = st.selectbox("Phone service", ["Yes", "No"], key=f"{key_prefix}_phone")
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"], key=f"{key_prefix}_lines"
        )
    with columns[1]:
        internet = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"], key=f"{key_prefix}_internet"
        )
        security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"], key=f"{key_prefix}_security"
        )
        backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"], key=f"{key_prefix}_backup"
        )
        device = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"], key=f"{key_prefix}_device"
        )
        support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"], key=f"{key_prefix}_support"
        )
        tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"], key=f"{key_prefix}_tv"
        )
        movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"], key=f"{key_prefix}_movies"
        )
    with columns[2 if len(columns) > 2 else 1]:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"], key=f"{key_prefix}_contract"
        )
        paperless = st.selectbox(
            "Paperless billing", ["Yes", "No"], key=f"{key_prefix}_paperless"
        )
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key=f"{key_prefix}_payment",
        )
        monthly = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0, value=75.0,
            step=1.0, key=f"{key_prefix}_monthly",
        )
        total = st.number_input(
            "Total charges ($)", min_value=0.0, max_value=10000.0, value=900.0,
            step=25.0, key=f"{key_prefix}_total",
        )

    return {
        "customerID": f"manual-{key_prefix}",
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


def render_executive(bundle: dict[str, Any], customers: pd.DataFrame) -> None:
    st.title("Customer churn intelligence")
    st.write(
        "A decision workspace for finding preventable churn, understanding the drivers, "
        "and turning model output into a defensible retention action."
    )
    best = bundle["metrics"][bundle["model_name"]]
    probabilities = get_portfolio_probabilities(bundle["model"], customers)
    risk_counts = pd.Series([risk_level(value) for value in probabilities]).value_counts()
    high_share = risk_counts.get("High", 0) / len(probabilities)
    a, b, c, d = st.columns(4)
    with a:
        _metric_card("Customers analyzed", f"{len(customers):,}")
    with b:
        _metric_card("Observed churn rate", f"{bundle['churn_rate']:.1%}")
    with c:
        _metric_card("High-risk coverage", f"{high_share:.1%}")
    with d:
        _metric_card("Best ROC-AUC", f"{best['roc_auc']:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk portfolio")
        risk_frame = pd.DataFrame(
            {"Risk level": ["Low", "Medium", "High"], "Customers": [risk_counts.get(level, 0) for level in ["Low", "Medium", "High"]]}
        )
        fig = px.bar(
            risk_frame, x="Risk level", y="Customers", color="Risk level",
            color_discrete_map={"Low": "#2f8f83", "Medium": "#d89b35", "High": "#d95f59"},
        )
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Churn by contract")
        contract = customers.groupby("Contract", as_index=False)["Churn"].mean()
        contract["Churn"] = contract["Churn"].map(lambda value: value * 100)
        fig = px.bar(contract, x="Contract", y="Churn", text_auto=".1f", labels={"Churn": "Churn rate (%)"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Where attention should go")
    portfolio = customers[["Contract", "tenure_bucket", "PaymentMethod", "MonthlyCharges", "Churn"]].copy()
    portfolio["Risk score"] = probabilities
    portfolio["Risk level"] = [risk_level(value) for value in probabilities]
    st.dataframe(
        portfolio.sort_values("Risk score", ascending=False).head(12),
        use_container_width=True,
        hide_index=True,
    )


def render_prediction(bundle: dict[str, Any]) -> None:
    st.title("Churn prediction")
    st.write("Score a customer, see the model's strongest drivers, and save the case plan.")
    with st.form("prediction_form"):
        values = customer_form("prediction", compact=False)
        submitted = st.form_submit_button("Score customer", type="primary")
    if submitted:
        render_prediction_result(bundle, values, log_result=True)
    else:
        st.info("Enter a customer profile above, then select Score customer.")


def render_what_if(bundle: dict[str, Any]) -> None:
    st.title("What-if analysis")
    st.write("Adjust customer attributes to explore which interventions could move risk.")
    values = customer_form("what_if", compact=False)
    if st.button("Analyse", type="primary"):
        render_prediction_result(bundle, values)
    else:
        st.info("Set customer attributes above, then select Analyse.")


def render_analytics(bundle: dict[str, Any], customers: pd.DataFrame) -> None:
    st.title("Customer analytics")
    st.write("Explore churn patterns across the source population and export a scored segment.")
    filters = st.columns(3)
    with filters[0]:
        contract_filter = st.multiselect(
            "Contract",
            sorted(customers["Contract"].unique()),
            default=sorted(customers["Contract"].unique()),
        )
    with filters[1]:
        tenure_filter = st.multiselect(
            "Tenure segment",
            sorted(customers["tenure_bucket"].unique()),
            default=sorted(customers["tenure_bucket"].unique()),
        )
    with filters[2]:
        payment_filter = st.multiselect(
            "Payment method",
            sorted(customers["PaymentMethod"].unique()),
            default=sorted(customers["PaymentMethod"].unique()),
        )
    charge_range = st.slider(
        "Monthly charge range",
        float(customers["MonthlyCharges"].min()),
        float(customers["MonthlyCharges"].max()),
        (
            float(customers["MonthlyCharges"].min()),
            float(customers["MonthlyCharges"].max()),
        ),
    )
    filtered = customers[
        customers["Contract"].isin(contract_filter)
        & customers["tenure_bucket"].isin(tenure_filter)
        & customers["PaymentMethod"].isin(payment_filter)
        & customers["MonthlyCharges"].between(*charge_range)
    ].copy()
    filtered["Churn probability"] = bundle["model"].predict_proba(
        filtered.drop(columns=["Churn", "customerID"])
    )[:, 1]
    filtered["Risk level"] = filtered["Churn probability"].map(risk_level)
    x, y = st.columns(2)
    with x:
        st.plotly_chart(
            px.scatter(
                filtered, x="tenure", y="MonthlyCharges", color="Risk level",
                color_discrete_map={"Low": "#2f8f83", "Medium": "#d89b35", "High": "#d95f59"},
                hover_data=["Contract", "PaymentMethod", "Churn probability"],
                title="Tenure vs monthly charges",
            ),
            use_container_width=True,
        )
    with y:
        by_payment = filtered.groupby("PaymentMethod", as_index=False)["Churn"].mean()
        by_payment["Churn"] = by_payment["Churn"] * 100
        st.plotly_chart(
            px.bar(by_payment, x="PaymentMethod", y="Churn", labels={"Churn": "Churn rate (%)"}, title="Churn by payment method"),
            use_container_width=True,
        )
    st.download_button(
        "Download scored segment",
        filtered.to_csv(index=False).encode("utf-8"),
        "scored_customer_segment.csv",
        "text/csv",
    )
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)


def render_performance(bundle: dict[str, Any]) -> None:
    st.title("Model performance")
    st.write("The model is selected on holdout ROC-AUC after cross-validated F1 tuning.")
    rows = []
    for name, values in bundle["metrics"].items():
        rows.append({"Model": name, **{key: values[key] for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1"]}})
    metrics_frame = pd.DataFrame(rows).set_index("Model")
    st.dataframe(metrics_frame.style.format("{:.3f}"), use_container_width=True)
    st.caption(
        f"Selected model: {bundle['model_name']}. Training rows: {bundle['train_rows']:,}; "
        f"test rows: {bundle['test_rows']:,}. {bundle['class_weighting']}"
    )
    left, right = st.columns(2)
    with left:
        roc_fig = go.Figure()
        for name, curve in bundle["roc_curves"].items():
            roc_fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], name=name, mode="lines"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", mode="lines", line=dict(dash="dash")))
        roc_fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", height=380)
        st.plotly_chart(roc_fig, use_container_width=True)
    with right:
        confusion = bundle["metrics"][bundle["model_name"]]["confusion_matrix"]
        st.plotly_chart(
            px.imshow(
                confusion, text_auto=True, color_continuous_scale="Tealgrn",
                labels=dict(x="Predicted", y="Observed", color="Customers"),
                x=["No churn", "Churn"], y=["No churn", "Churn"], title="Best-model confusion matrix",
            ),
            use_container_width=True,
        )
    with st.expander("Feature importance comparison"):
        importance = pd.DataFrame(bundle["feature_importance"])
        st.plotly_chart(
            px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", title="Built-in importance (secondary view)"),
            use_container_width=True,
        )
        if st.button("Calculate global SHAP importance"):
            with st.spinner("Calculating SHAP values on a representative sample..."):
                global_values = global_shap_importance(bundle["model"], get_customers().drop(columns=["Churn", "customerID"]))
            st.dataframe(global_values, use_container_width=True, hide_index=True)


def render_history() -> None:
    st.title("Prediction history")
    st.write("Every saved manual score is logged in SQLite with its probability and recommendation.")
    history = read_history()
    if history.empty:
        st.info("No saved predictions yet. Score a customer to create the first history record.")
        return
    a, b = st.columns(2)
    with a:
        st.metric("Saved predictions", f"{len(history):,}")
    with b:
        st.metric("Average predicted churn", f"{history['churn_probability'].mean():.1%}")
    st.plotly_chart(
        px.histogram(history, x="churn_probability", color="risk_level", nbins=12, title="Saved probability distribution"),
        use_container_width=True,
    )
    st.download_button(
        "Download history",
        history.to_csv(index=False).encode("utf-8"),
        "prediction_history.csv",
        "text/csv",
    )
    st.dataframe(history, use_container_width=True, hide_index=True)
    with st.expander("View recommendation details"):
        selected_id = st.selectbox("Prediction record", history["id"].tolist())
        row = history.loc[history["id"] == selected_id].iloc[0]
        st.json(json.loads(row["recommendation_json"]))


def main() -> None:
    if "bundle" not in st.session_state:
        with st.spinner("Loading trained churn intelligence..."):
            st.session_state["bundle"] = get_bundle()
            st.session_state["customers"] = get_customers()
    bundle = st.session_state["bundle"]
    customers = st.session_state["customers"]
    st.sidebar.title("Retent")
    st.sidebar.caption("Churn risk intelligence")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Dashboard",
            "Churn Prediction",
            "What-If Analysis",
            "Customer Analytics",
            "Model Performance",
            "Prediction History",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption(f"Model: {bundle['model_name']}")
    st.sidebar.caption(f"Dataset: {bundle['dataset_source']}")

    if page == "Executive Dashboard":
        render_executive(bundle, customers)
    elif page == "Churn Prediction":
        render_prediction(bundle)
    elif page == "What-If Analysis":
        render_what_if(bundle)
    elif page == "Customer Analytics":
        render_analytics(bundle, customers)
    elif page == "Model Performance":
        render_performance(bundle)
    else:
        render_history()


if __name__ == "__main__":
    main()
