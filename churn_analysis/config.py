from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
DB_PATH = ROOT_DIR / "churn_history.db"

DATA_URL = (
    "https://github.com/IBM/telco-customer-churn-on-icp4d/raw/"
    "refs/heads/master/data/Telco-Customer-Churn.csv"
)

RANDOM_STATE = 42
HIGH_RISK_THRESHOLD = 0.65
MEDIUM_RISK_THRESHOLD = 0.35

PREDICTION_INPUT_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
