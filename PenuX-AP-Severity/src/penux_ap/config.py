"""Configuration constants for penux_ap."""
from pathlib import Path

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
DEFAULT_THRESHOLD: float = 0.5
DEFAULT_N_BOOTSTRAPS: int = 1000

SUPPORTED_EXTENSIONS: set[str] = {".csv", ".xlsx", ".xls"}

IDENTIFIER_COLUMN_PATTERNS: list[str] = [
    "name", "patient_name", "patientname",
    "patient_id", "patientid", "pid",
    "hospital_id", "hospitalid",
    "mrn", "medical_record_number",
    "id no", "id_no", "id no.", "identifier", "identity",
    "phone", "mobile", "cell", "cellphone", "tel", "telephone", "fax",
    "address", "email",
    "passport", "national_id", "social_security",
    "insurance", "姓名", "序号", "住院号", "身份证",
]

LIKELY_TARGET_COLUMNS: list[str] = [
    "severe",
    "severity",
    "sap",
    "target",
    "label",
    "outcome",
    "class",
    "severe_ap",
    "severe_acute_pancreatitis",
]

RISK_THRESHOLDS: dict[str, float] = {
    "low": 0.2,
    "intermediate": 0.5,
}

MODEL_REGISTRY_FLAGS: dict[str, bool] = {
    "xgboost": True,
    "lightgbm": True,
    "mlp": True,
}

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
