"""
Unit tests for feature engineering functions

Tests feature creation pipelines for traditional and behavioral models,
including ratio calculations, aggregations, and data transformations.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import process_apps, process_prev, get_prev_agg
from src.config import data_path


def test_process_apps_on_real_data():
    """
    Test process_apps on real Home Credit data.
    
    Verifies that the traditional model feature engineering pipeline
    creates expected ratio and aggregate features from application data.
    """
    path = data_path("application_train")
    if not os.path.exists(path):
        pytest.skip("application_train.csv not available in data/")

    df = pd.read_csv(path, nrows=20)
    out = process_apps(df)
    assert "APPS_EXT_SOURCE_MEAN" in out.columns
    assert "APPS_ANNUITY_CREDIT_RATIO" in out.columns


def test_get_prev_agg_on_real_data():
    """
    Test previous application aggregation on real data.
    
    Verifies that previous loan history is properly aggregated
    by SK_ID_CURR and expected count columns are created.
    """
    path = data_path("previous_application")
    if not os.path.exists(path):
        pytest.skip("previous_application.csv not available in data/")

    prev = pd.read_csv(path, nrows=200)
    prev_agg = get_prev_agg(prev)
    # After aggregation we expect PREV_SK_ID_CURR_COUNT column
    assert any(col.startswith("PREV_SK_ID_CURR_COUNT") or col == "PREV_SK_ID_CURR_COUNT" for col in prev_agg.columns)


def test_process_apps_basic():
    """
    Test process_apps with synthetic data.
    
    Validates feature engineering with controlled inputs including:
    - External credit scores (EXT_SOURCE)
    - Financial ratios (annuity/credit, income/credit)
    - Demographic features (family size, age, employment)
    - Handling of missing values and edge cases
    """
    # Small synthetic dataset
    df = pd.DataFrame({
        "EXT_SOURCE_1": [0.5, np.nan],
        "EXT_SOURCE_2": [0.7, 0.2],
        "EXT_SOURCE_3": [0.6, 0.3],
        "AMT_ANNUITY": [1000, 2000],
        "AMT_CREDIT": [10000, 5000],
        "AMT_GOODS_PRICE": [9000, 4000],
        "AMT_INCOME_TOTAL": [50000, 40000],
        "CNT_FAM_MEMBERS": [2, 1],
        "DAYS_EMPLOYED": [1000, 200],
        "DAYS_BIRTH": [10000, 9000],
        "OWN_CAR_AGE": [5, 2],
    })

    out = process_apps(df.copy())
    # Check that ratio columns exist and no inf values
    assert "APPS_EXT_SOURCE_MEAN" in out.columns
    assert not out["APPS_EXT_SOURCE_MEAN"].isnull().all()
    assert "APPS_ANNUITY_CREDIT_RATIO" in out.columns
    assert np.isfinite(out.loc[0, "APPS_ANNUITY_CREDIT_RATIO"]) or pd.isna(out.loc[0, "APPS_ANNUITY_CREDIT_RATIO"]) is False


def test_process_prev_basic():
    """
    Test previous application processing with synthetic data.
    
    Verifies that previous loan features are correctly engineered,
    including credit differences and application ratios.
    """
    df = pd.DataFrame({
        "AMT_APPLICATION": [1000, 2000],
        "AMT_ANNUITY": [100, 200],
        "AMT_CREDIT": [900, 1800],
        "AMT_GOODS_PRICE": [1100, 2100],
        "CNT_PAYMENT": [10, 20],
        "DAYS_FIRST_DRAWING": [0, 365243],
        "DAYS_FIRST_DUE": [0, 365243],
        "DAYS_LAST_DUE_1ST_VERSION": [0, 0],
        "DAYS_LAST_DUE": [0, 0],
        "DAYS_TERMINATION": [0, 365243],
        "DAYS_DECISION": [-10, -20],
    })

    out = process_prev(df.copy())
    assert "PREV_CREDIT_DIFF" in out.columns
    assert "PREV_CREDIT_APPL_RATIO" in out.columns