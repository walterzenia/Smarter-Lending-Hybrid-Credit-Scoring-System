import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc

models_dir = Path('models')
paths = sorted(models_dir.glob('*.pkl')) + sorted(models_dir.glob('*.joblib'))
if not paths:
    print('No model files found in models/')
    raise SystemExit(1)

holdout_path = Path('data/smoke_engineered.csv')
if not holdout_path.exists():
    print('Holdout not found at data/smoke_engineered.csv')
    raise SystemExit(1)

df = pd.read_csv(holdout_path)
if 'TARGET' not in df.columns and 'target' not in df.columns:
    print('Holdout does not contain TARGET/target column')
    raise SystemExit(1)

y_col = 'TARGET' if 'TARGET' in df.columns else 'target'
# sample for speed
n = min(20000, len(df))
df_sample = df.sample(n, random_state=42)
X_full = df_sample.drop(columns=[y_col], errors='ignore')
y = df_sample[y_col].values
# drop rows with missing labels
mask = ~pd.isna(y)
if not mask.all():
    print(f'Dropped {len(mask) - int(mask.sum())} rows with missing TARGET')
    X_full = X_full.loc[mask]
    y = y[mask]

for p in paths:
    print('\nModel:', p)
    try:
        m = joblib.load(p)
        print('  Loaded type:', type(m))
        # try to determine expected feature names
        expected = None
        if hasattr(m, 'feature_names_in_'):
            expected = list(m.feature_names_in_)
        elif hasattr(m, 'named_steps'):
            for step in m.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    expected = list(step.feature_names_in_)
                    break
        if expected is None:
            expected = list(X_full.columns)
        # align
        missing = [c for c in expected if c not in X_full.columns]
        Xm = X_full.copy()
        for c in missing:
            Xm[c] = np.nan
        Xm = Xm[expected]
        # Basic defensives: replace infs, encode categoricals, fill missing values to give estimator a chance
        Xm = Xm.replace([np.inf, -np.inf], np.nan)
        for col in Xm.select_dtypes(include=['object', 'category']).columns:
            Xm[col] = Xm[col].astype('category').cat.codes
        # Fill remaining NaNs with column medians where possible, then zeros
        try:
            Xm = Xm.fillna(Xm.median(numeric_only=True)).fillna(0)
        except Exception:
            Xm = Xm.fillna(0)

        proba = None
        try:
            ptmp = m.predict_proba(Xm)
            if ptmp.shape[1] >= 2:
                proba = ptmp[:, 1]
            else:
                proba = ptmp[:, 0]
        except Exception as e:
            print('  predict_proba failed:', e)
            try:
                proba = m.decision_function(Xm)
            except Exception as e2:
                print('  decision_function failed:', e2)
                proba = None
        if proba is None:
            print('  Model has no probability/decision outputs. Skipping AUC')
            continue
        fpr, tpr, _ = roc_curve(y, proba)
        auc_score = auc(fpr, tpr)
        print(f'  AUC on sample ({len(y)} rows): {auc_score:.4f}')
    except Exception as e:
        print('  Failed to evaluate model:')
        import traceback
        traceback.print_exc()
