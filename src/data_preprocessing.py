"""
Data Loading Pipeline for Loan Default Prediction System

Handles loading and initial validation of Home Credit Bureau datasets.
Supports multiple data sources including:
- Application data (train/test)
- Previous loan applications
- Bureau credit history
- Balance histories (POS, installments, credit cards)

All file paths are managed through centralized config for easy deployment.

Date: November 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import os
import logging
from .config import data_path

logger = logging.getLogger(__name__)


def get_balance_data():
    """
    Load balance-related CSV files from the data directory.
    
    Loads three types of balance history:
    1. POS (Point of Sale) cash balance
    2. Installment payments history
    3. Credit card balance history
    
    Returns:
    --------
    tuple of pd.DataFrame
        (pos, installments, credit_card) DataFrames
        
    Raises:
    -------
    FileNotFoundError
        If any required data file is missing from data/ directory
        
    Example:
    --------
    >>> pos, ins, card = get_balance_data()
    >>> print(f"POS records: {len(pos)}, Installments: {len(ins)}, Card: {len(card)}")
    """
    pos_path = data_path("POS_CASH_balance")
    ins_path = data_path("installments_payments")
    card_path = data_path("credit_card_balance")

    for p in (pos_path, ins_path, card_path):
        if not os.path.exists(p):
            logger.error("Required data file not found: %s", p)
            raise FileNotFoundError(f"Required data file not found: {p}")

    pos = pd.read_csv(pos_path)
    ins = pd.read_csv(ins_path)
    card = pd.read_csv(card_path)

    return pos, ins, card


def get_dataset():
    """
    Load all Home Credit datasets and return as tuple.
    
    Combines training and test application data, then loads all supporting
    datasets for comprehensive credit risk modeling.
    
    Data Sources Loaded:
    --------------------
    1. application_train.csv: Training set with TARGET labels
    2. application_test.csv: Test set without labels
    3. previous_application.csv: Historical loan applications
    4. bureau.csv: Credit bureau data (external credit history)
    5. bureau_balance.csv: Monthly balances from bureau
    6. POS_CASH_balance.csv: Point-of-sale installment data
    7. installments_payments.csv: Payment history
    8. credit_card_balance.csv: Monthly credit card data
    
    Returns:
    --------
    tuple of pd.DataFrame
        (apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
        
        - apps: Combined train+test applications (307,511 rows)
        - prev: Previous applications (1,670,214 rows)
        - bureau: Bureau credit data (1,716,428 rows)
        - bureau_bal: Bureau balance history (27,299,925 rows)
        - pos_bal: POS cash balance (10,001,358 rows)
        - install: Installments payments (13,605,401 rows)
        - card_bal: Credit card balance (3,840,312 rows)
    
    Data Pipeline:
    --------------
    1. Verify all files exist using config.data_path()
    2. Load CSVs with pandas (uses Git LFS for large files)
    3. Concatenate train/test for unified processing
    4. Return tuple for downstream feature engineering
    
    Raises:
    -------
    FileNotFoundError
        If any required CSV file is missing from data/ directory
        
    Memory Note:
    ------------
    Full dataset requires ~5GB RAM. For limited memory environments:
    - Use nrows parameter in pd.read_csv()
    - Process datasets in chunks
    - Filter to specific SK_ID_CURR ranges
        
    Example:
    --------
    >>> apps, prev, bureau, bb, pos, ins, card = get_dataset()
    >>> print(f"Total applications: {len(apps)}")
    >>> print(f"Labeled: {apps['TARGET'].notna().sum()}")
    Total applications: 307511
    Labeled: 307511
    
    Integration:
    ------------
    This function feeds into the feature engineering pipeline:
    >>> apps, prev, bureau, bb, pos, ins, card = get_dataset()
    >>> apps_fe = process_apps(apps)  # Feature engineering
    >>> prev_fe = process_prev(prev)
    >>> # ... continue with bureau, balance aggregations
    """

    # Read training and test files (train/test were previously swapped)
    app_train_path = data_path("application_train")
    app_test_path = data_path("application_test")

    if not os.path.exists(app_train_path) or not os.path.exists(app_test_path):
        logger.error("Application train/test files not found: %s or %s", app_train_path, app_test_path)
        raise FileNotFoundError("Application train/test files not found in data directory")

    app_train = pd.read_csv(app_train_path)
    app_test = pd.read_csv(app_test_path)
    apps = pd.concat([app_train, app_test], ignore_index=True)

    prev_path = data_path("previous_application")
    bureau_path = data_path("bureau")
    bureau_bal_path = data_path("bureau_balance")

    for p in (prev_path, bureau_path, bureau_bal_path):
        if not os.path.exists(p):
            logger.error("Required data file not found: %s", p)
            raise FileNotFoundError(f"Required data file not found: {p}")

    prev = pd.read_csv(prev_path)
    bureau = pd.read_csv(bureau_path)
    bureau_bal = pd.read_csv(bureau_bal_path)

    pos_bal, install, card_bal = get_balance_data()

    return apps, prev, bureau, bureau_bal, pos_bal, install, card_bal
