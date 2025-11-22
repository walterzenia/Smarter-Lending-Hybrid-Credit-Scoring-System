import joblib
from pathlib import Path
import traceback

models_dir = Path('models')
paths = sorted(models_dir.glob('*.pkl')) + sorted(models_dir.glob('*.joblib'))
if not paths:
    print('No model files found in models/')
for p in paths:
    print('\nFile:', p)
    try:
        m = joblib.load(p)
        print('  Loaded type:', type(m))
        print('  predict_proba:', hasattr(m, 'predict_proba'))
        print('  decision_function:', hasattr(m, 'decision_function'))
        try:
            # try introspecting pipeline
            from sklearn.pipeline import Pipeline
            if isinstance(m, Pipeline):
                print('  Detected sklearn Pipeline; named steps:', list(m.named_steps.keys()))
        except Exception:
            pass
    except Exception as e:
        print('  Failed to load model:')
        traceback.print_exc()

holdout = Path('data/smoke_engineered.csv')
print('\nHoldout exists:', holdout.exists())
