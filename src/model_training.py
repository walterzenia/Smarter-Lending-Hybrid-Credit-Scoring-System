import argparse
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sys
from sklearn.preprocessing import FunctionTransformer
from src.extract_features import traditional_features, hybrid_features, behaviorial_features
from src.data_preprocessing import get_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# /c:/Users/user/Desktop/Loan Default Hybrid System/src/model_training.py
"""
Train a classifier on feature columns produced by extract_features.py.

Usage example:
    python model_training.py --input features.csv --features all --target defaulted 
        --model random_forest --output model.pkl --test-size 0.2 --random-state 42
"""

from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
)

def parse_args():
    p = argparse.ArgumentParser(description="Train a model from one or more feature CSVs.")
    # Mutually exclusive: either pass precomputed feature CSVs via --input, or request raw-data preprocessing
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        "-i",
        nargs="+",
        help="One or more CSV files with precomputed features (from extract_features.py).",
    )
    group.add_argument(
        "--preprocess-mode",
        dest="preprocess_mode",
        choices=["traditional", "hybrid", "behaviorial"],
        help="Run feature engineering from raw data in data/ using the selected mode.",
    )
    p.add_argument(
        "--features",
        "-f",
        default="all",
        help="Comma-separated feature names or 'all' to use all columns except target",
    )
    p.add_argument("--target", "-t", default="TARGET", help="Name of target column in CSV")
    p.add_argument(
        "--save-features",
        "-s",
        dest="save_features",
        default=None,
        help="Optional path to write engineered/loaded features CSV before training",
    )
    p.add_argument(
        "--model",
        "-m",
        choices=["logistic", "random_forest", "lgbm", "xgboost"],
        default="logistic",
        help="Model type to train",
    )
    p.add_argument("--output", "-o", default="model.pkl", help="Path to save trained model pipeline (joblib)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()


def build_pipeline(X, model_type="lgbm", preprocess_mode=None, random_state=42):
    """
    Build an sklearn Pipeline given:
      - X: a pandas DataFrame OR a list/tuple of DataFrames (used to infer column types when doing column-based preprocessing
           or passed as-is to preprocess_mode feature functions)
      - model_type: string, one of ("xgboost","lgbm","random_forest","logistic"); defaults to lgbm
      - preprocess_mode: None or "traditional" / "hybrid" / "behaviorial"
    If preprocess_mode is provided, the corresponding feature function (imported from src.extract_features)
    will receive the same X value passed to build_pipeline (so it can be a list of DataFrames).
    Otherwise a column-based ColumnTransformer is created using the first DataFrame to infer dtypes.
    """
    # Determine if X is a list of DataFrames or a single DataFrame
    is_multi = isinstance(X, (list, tuple))
    sample_df = X[0] if is_multi else X

    # Choose preprocessing transformer
    if preprocess_mode is not None:
        if preprocess_mode == "traditional":
            feature_fn = traditional_features
        elif preprocess_mode == "hybrid":
            feature_fn = hybrid_features
        elif preprocess_mode == "behaviorial":
            feature_fn = behaviorial_features
        else:
            raise ValueError(f"Unknown preprocess_mode: {preprocess_mode}")

        # Wrap feature function so it can be used inside an sklearn Pipeline.
        # The feature functions sometimes expect multiple dataframes (apps, prev, bureau, ...).
        def _wrapped_feature_fn(x):
            # If a list/tuple of DataFrames was passed, unpack it into the feature function
            if isinstance(x, (list, tuple)):
                return feature_fn(*x)
            # If a single DataFrame is passed, try calling feature_fn(df); if that fails, raise helpful error
            try:
                return feature_fn(x)
            except TypeError as e:
                raise ValueError(
                    "preprocess_mode feature function requires multiple DataFrames. "
                    "Pass X as a list/tuple of DataFrames when using preprocess_mode."
                ) from e

        preprocessor = FunctionTransformer(_wrapped_feature_fn, validate=False)

    else:
        # Column-based preprocessing: numeric -> impute+scale, categorical -> impute+onehot
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        transformers = []
        if numeric_cols:
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipeline, numeric_cols))

        if categorical_cols:
            # OneHotEncoder changed its parameter name across sklearn versions
            try:
                _ = OneHotEncoder(handle_unknown="ignore", sparse=False)
                onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                # newer sklearn uses 'sparse_output' instead of 'sparse'
                onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                    ("onehot", onehot),
                ]
            )
            transformers.append(("cat", cat_pipeline, categorical_cols))

        preprocessor = ColumnTransformer(transformers=transformers) if transformers else "passthrough"

    # Choose classifier
    model_type = (model_type or "").lower()
    if model_type == "xgboost":
        clf = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss")
    elif model_type in ("lgbm", "lightgbm"):
        clf = LGBMClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    elif model_type == "logistic":
        clf = LogisticRegression(max_iter=2000, random_state=random_state)
    else:
        # default fallback
        clf = LGBMClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    return pipeline

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("model_training")

    args = parse_args()

    # If preprocess_mode is provided, load raw datasets and compute features using extract_features
    if args.preprocess_mode:
        apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = get_dataset()

        if args.preprocess_mode == "traditional":
            features_df = traditional_features(apps, prev, bureau, bureau_bal)
        elif args.preprocess_mode == "hybrid":
            features_df = hybrid_features(apps, bureau, bureau_bal, prev, pos_bal, install, card_bal)
        elif args.preprocess_mode == "behaviorial":
            features_df = behaviorial_features(apps, pos_bal, install, card_bal)
        else:
            raise SystemExit(f"Unknown preprocess_mode: {args.preprocess_mode}")

        # Ensure target exists and drop rows without a target (e.g., test rows)
        if args.target not in features_df.columns:
            logger.error("Target column '%s' not found in engineered features", args.target)
            raise SystemExit(f"Target column '{args.target}' not found in engineered features")

        df = features_df.dropna(subset=[args.target])

        if args.features.strip().lower() == "all":
            feature_cols = [c for c in df.columns if c != args.target]
        else:
            feature_cols = [f.strip() for f in args.features.split(",") if f.strip()]
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise SystemExit(f"Feature columns not found in engineered features: {missing}")

        if not feature_cols:
            raise SystemExit("No feature columns selected.")

        # Optionally save features for debugging
        if args.save_features:
            logger.info("Saving engineered features to %s", args.save_features)
            df.loc[:, feature_cols + [args.target]].to_csv(args.save_features, index=False)

        X = df[feature_cols]
        y = df[args.target]

    else:
        # args.input may be a list of file paths; concat them into a single DataFrame
        if isinstance(args.input, (list, tuple)) and len(args.input) > 1:
            df = pd.concat([pd.read_csv(p) for p in args.input], ignore_index=True)
        else:
            df = pd.read_csv(args.input[0] if isinstance(args.input, (list, tuple)) else args.input)

        if args.target not in df.columns:
            raise SystemExit(f"Target column '{args.target}' not found in {args.input}")

        if args.features.strip().lower() == "all":
            feature_cols = [c for c in df.columns if c != args.target]
        else:
            feature_cols = [f.strip() for f in args.features.split(",") if f.strip()]
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise SystemExit(f"Feature columns not found in CSV: {missing}")

        if not feature_cols:
            raise SystemExit("No feature columns selected.")

        # Optionally save loaded features CSV
        if args.save_features:
            logger.info("Saving loaded features to %s", args.save_features)
            df.loc[:, feature_cols + [args.target]].to_csv(args.save_features, index=False)

        X = df[feature_cols]
        y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    logger.info("Building pipeline with model=%s preprocess_mode=%s", args.model, args.preprocess_mode)
    pipeline = build_pipeline(X_train, args.model, preprocess_mode=None, random_state=args.random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # If binary, try to print ROC AUC (requires predict_proba)
    try:
        if len(np.unique(y)) == 2 and hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probs = pipeline.predict_proba(X_test)[:, 1]
            print("ROC AUC:", round(roc_auc_score(y_test, probs), 4))
    except Exception:
        pass

    joblib.dump(pipeline, args.output)
    logger.info("Saved model pipeline to %s", args.output)
    print(f"Saved model pipeline to {args.output}")

if __name__ == "__main__":
    main()