from pathlib import Path

# Project root (two levels up from this file: src/ -> project root)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Filenames used by the project
FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "previous_application": "previous_application.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "POS_CASH_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}


def data_path(name: str):
    """Return full path for a data file key from FILES or a direct filename.

    Examples:
        data_path('application_train') -> Path(.../data/application_train.csv)
        data_path('some_other.csv') -> Path(.../data/some_other.csv)
    """
    if name in FILES:
        return DATA_DIR / FILES[name]
    return DATA_DIR / name


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
