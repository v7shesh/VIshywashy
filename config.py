"""Central configuration: paths and constants shared across the pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

RAW_CSV_PATH = DATA_DIR / "dataset.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"

TARGET_COL = "Target"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

# --- Feature groups (domain knowledge of the UCI/Kaggle dropout dataset) ---
# Nominal categorical codes with no inherent order -> one-hot encoded.
NOMINAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Already-binary 0/1 flags -> passed through the numeric (scaling) branch.
BINARY_FEATURES = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Continuous / count numeric features -> scaled.
NUMERIC_FEATURES = [
    "Application order",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Features most predictive of dropout risk, used to drive the EDA and the
# per-student "top risk factors" explanation in intervention.py.
KEY_RISK_FEATURES = [
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
    "Age at enrollment",
]

# Risk bands for P(Dropout) -> intervention routing (see src/intervention.py)
RISK_BAND_THRESHOLDS = {
    "Low": (0.0, 0.33),
    "Medium": (0.33, 0.66),
    "High": (0.66, 1.01),
}

for _d in (DATA_DIR, MODELS_DIR, DOCS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
