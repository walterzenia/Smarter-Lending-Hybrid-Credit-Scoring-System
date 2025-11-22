"""Smoke test: run light end-to-end check using real data in data/.

This script will:
 - load raw datasets via `get_dataset()`
 - run the `hybrid` feature engineering pipeline
 - save engineered features to `data/smoke_engineered.csv`
 - sample up to 5000 rows (rows that have the target) and train a small model to verify pipeline

Run from repo root with the project's venv active.
"""
import logging
from pathlib import Path
import sys
import pandas as pd

# Ensure project root is on sys.path so `src` package imports work when script is executed directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preprocessing import get_dataset
from src.extract_features import hybrid_features
from src.config import data_path
from src.model_training import build_pipeline
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("smoke_test")

    try:
        apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = get_dataset()
    except FileNotFoundError as e:
        logger.error("Data files missing: %s", e)
        sys.exit(2)

    # To keep the smoke test memory/lightweight, sample a subset of SK_ID_CURR from apps
    sample_n = 20000
    n_available = len(apps)
    sample_n = min(sample_n, n_available)
    logger.info("Sampling %d rows from apps (available %d)", sample_n, n_available)
    sampled_ids = apps["SK_ID_CURR"].sample(sample_n, random_state=42).unique()

    apps_small = apps[apps["SK_ID_CURR"].isin(sampled_ids)].copy()
    prev_small = prev[prev["SK_ID_CURR"].isin(sampled_ids)].copy()
    bureau_small = bureau[bureau["SK_ID_CURR"].isin(sampled_ids)].copy()
    # bureau_bal references SK_ID_BUREAU; filter by SK_ID_BUREAU present in the sampled bureau rows
    bureau_bureau_ids = bureau_small["SK_ID_BUREAU"].unique() if "SK_ID_BUREAU" in bureau_small.columns else []
    bureau_bal_small = bureau_bal[bureau_bal.get("SK_ID_BUREAU", pd.Series()).isin(bureau_bureau_ids)].copy()
    pos_bal_small = pos_bal[pos_bal["SK_ID_CURR"].isin(sampled_ids)].copy()
    install_small = install[install["SK_ID_CURR"].isin(sampled_ids)].copy()
    card_bal_small = card_bal[card_bal["SK_ID_CURR"].isin(sampled_ids)].copy()

    logger.info("Running hybrid feature engineering on sampled subset... this may take a while depending on dataset size")
    features_df = hybrid_features(apps_small, bureau_small, bureau_bal_small, prev_small, pos_bal_small, install_small, card_bal_small)

    out_path = data_path("smoke_engineered.csv")
    logger.info("Saving engineered features to %s", out_path)
    features_df.to_csv(out_path, index=False)

    target_col = "TARGET"
    if target_col not in features_df.columns:
        logger.warning("Target column '%s' not present in engineered features; smoke test finished after saving features.", target_col)
        print(f"Saved engineered features to {out_path}")
        return

    df_with_target = features_df.dropna(subset=[target_col])
    if df_with_target.empty:
        logger.warning("No rows with a target value after engineering. Exiting.")
        return

    n = min(5000, len(df_with_target))
    sample = df_with_target.sample(n, random_state=42) if len(df_with_target) > n else df_with_target

    feature_cols = [c for c in sample.columns if c != target_col]
    X = sample[feature_cols]
    y = sample[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Replace any infinite values introduced during feature engineering with NaN so sklearn imputers can handle them
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info("Building a quick pipeline and training on sample of %d rows", len(X_train))
    pipeline = build_pipeline(X_train, model_type="logistic")
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    logger.info("Smoke test model accuracy on sample: %.4f", score)
    print("Smoke test completed. Accuracy on sample: ", round(score, 4))


if __name__ == "__main__":
    main()
