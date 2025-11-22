"""
Ensemble Hybrid Model Class
Wrapper for combining traditional and behavioral models
"""
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', message='.*number of features.*')
warnings.filterwarnings('ignore', message='.*predict_disable_shape_check.*')

class EnsembleHybridModel:
    """Wrapper class for easy ensemble prediction"""
    def __init__(self, meta_model, model_trad, model_behav, trad_feats, behav_feats):
        self.meta_model = meta_model
        self.model_traditional = model_trad
        self.model_behavioral = model_behav
        self.traditional_features = trad_feats
        self.behavioral_features = behav_feats
        
        # Store key features used in meta-model training
        self.key_traditional = trad_feats[:10] if len(trad_feats) >= 10 else trad_feats
        self.key_behavioral = behav_feats[:10] if len(behav_feats) >= 10 else behav_feats
    
    def predict_proba(self, X):
        """Predict probabilities using the ensemble"""
        # Prepare features for each base model
        X_trad = X[self.traditional_features].copy()
        X_behav = X[self.behavioral_features].copy()
        
        # Handle missing and categorical
        for col in X_trad.columns:
            if X_trad[col].dtype in ['object', 'category']:
                X_trad[col] = X_trad[col].fillna('MISSING')
                le = LabelEncoder()
                X_trad[col] = le.fit_transform(X_trad[col].astype(str))
            else:
                X_trad[col] = X_trad[col].fillna(X_trad[col].median())
        
        for col in X_behav.columns:
            if X_behav[col].dtype in ['object', 'category']:
                X_behav[col] = X_behav[col].fillna('MISSING')
                le = LabelEncoder()
                X_behav[col] = le.fit_transform(X_behav[col].astype(str))
            else:
                X_behav[col] = X_behav[col].fillna(X_behav[col].median())
        
        # Get base model predictions
        try:
            pred_trad = self.model_traditional.predict_proba(X_trad)[:, 1]
        except:
            pred_trad = np.zeros(len(X))
        
        try:
            pred_behav = self.model_behavioral.predict_proba(X_behav)[:, 1]
        except:
            pred_behav = np.zeros(len(X))
        
        # Create meta-features
        meta_X = pd.DataFrame({
            'pred_traditional': pred_trad,
            'pred_behavioral': pred_behav,
            'pred_avg': (pred_trad + pred_behav) / 2,
            'pred_max': np.maximum(pred_trad, pred_behav),
            'pred_min': np.minimum(pred_trad, pred_behav),
            'pred_diff': np.abs(pred_trad - pred_behav),
            'pred_ratio': pred_trad / (pred_behav + 0.001)
        })
        
        # Add key features from both models (same as training)
        for feat in self.key_traditional:
            if feat in X_trad.columns:
                meta_X[f'trad_{feat}'] = X_trad[feat].values
        
        for feat in self.key_behavioral:
            if feat in X_behav.columns:
                meta_X[f'behav_{feat}'] = X_behav[feat].values
        
        # Get final prediction from meta-model
        final_proba = self.meta_model.predict(meta_X, num_iteration=self.meta_model.best_iteration)
        
        return np.column_stack([1 - final_proba, final_proba])
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
