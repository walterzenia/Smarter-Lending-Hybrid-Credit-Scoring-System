"""
Utility functions for the Loan Default Prediction Dashboard
"""
import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

# Add src to path for ensemble_model import
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

@st.cache_resource
def load_model(model_path):
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data(file_path):
    """Load dataset from CSV."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_available_models():
    """Get list of available model files from models/ directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    # Only show these 3 specific models
    target_models = [
        'model_hybrid.pkl',          # Traditional model
        'first_lgbm_model.pkl',      # Behavioral model
        'model_ensemble_wrapper.pkl' # Ensemble model
    ]
    
    available_models = []
    for model_name in target_models:
        model_path = models_dir / model_name
        if model_path.exists():
            available_models.append(str(model_path))
    
    return available_models

def get_model_type(model_name):
    """Determine model type from filename."""
    model_name = model_name.lower()
    if 'ensemble' in model_name or 'wrapper' in model_name:
        return 'ensemble'
    elif 'hybrid' in model_name:
        return 'traditional'
    elif 'first_lgbm' in model_name or 'behav' in model_name:
        return 'behavioral'
    else:
        return 'unknown'

def align_features(X, model):
    """Align input features to match model's expected features."""
    try:
        # Make a copy to avoid modifying original
        X = X.copy()
        
        # Convert categorical columns to numeric
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Try label encoding
            if X[col].dtype == 'object':
                # Simple label encoding (convert to category codes)
                X[col] = pd.Categorical(X[col]).codes
                # Replace -1 (missing category) with NaN
                X[col] = X[col].replace(-1, np.nan)
        
        # Get expected features
        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
        elif hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    expected = list(step.feature_names_in_)
                    break
            else:
                expected = list(X.columns)
        else:
            expected = list(X.columns)
        
        # Add missing columns with appropriate fill values
        missing = [c for c in expected if c not in X.columns]
        if len(missing) > 0:
            print(f"[ALIGN_FEATURES] Adding {len(missing)} missing features with default values (0)")
            # Use dict to create all missing columns at once (avoids fragmentation warnings)
            missing_dict = {c: 0 for c in missing}
            missing_df = pd.DataFrame(missing_dict, index=X.index)
            X = pd.concat([X, missing_df], axis=1)
        
        # Keep only expected columns
        X = X[expected]
        
        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill numeric columns with 0 (for manual input forms)
        for col in X.columns:
            if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if X[col].isnull().any():
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X[col] = X[col].fillna(median_val)
        
        return X
    except Exception as e:
        st.warning(f"Feature alignment issue: {e}")
        return X

def get_predictions(model, X):
    """Get predictions and probabilities from model."""
    try:
        # Check if this is an ensemble wrapper (has custom predict_proba method)
        is_ensemble = hasattr(model, 'model_traditional') and hasattr(model, 'model_behavioral')
        
        if is_ensemble:
            # Ensemble model handles its own feature alignment
            predictions = model.predict(X)
            proba = model.predict_proba(X)
            if proba.shape[1] >= 2:
                probabilities = proba[:, 1]
            else:
                probabilities = proba[:, 0]
        else:
            # Regular model - use standard alignment
            X_aligned = align_features(X, model)
            
            predictions = model.predict(X_aligned)
            
            probabilities = None
            try:
                proba = model.predict_proba(X_aligned)
                if proba.shape[1] >= 2:
                    probabilities = proba[:, 1]
                else:
                    probabilities = proba[:, 0]
            except:
                try:
                    probabilities = model.decision_function(X_aligned)
                except:
                    probabilities = None
        
        return predictions, probabilities
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def classify_risk(probability):
    """Classify risk level based on default probability."""
    if probability < 0.3:
        return "Low Risk 🟢"
    elif probability < 0.6:
        return "Medium Risk 🟡"
    else:
        return "High Risk 🔴"

def plot_distribution(df, column, title="Distribution"):
    """Plot distribution of a numeric column."""
    fig = px.histogram(df, x=column, title=title, 
                       marginal="box", 
                       color_discrete_sequence=['#636EFA'])
    fig.update_layout(
        template='plotly_white',
        height=400
    )
    return fig

def plot_correlation_heatmap(df, figsize=(12, 8)):
    """Plot correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, 
                cbar_kws={"shrink": 0.8},
                annot=False, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, pad=20)
    return fig

def plot_target_distribution(df, target_col='TARGET'):
    """Plot target variable distribution."""
    if target_col not in df.columns:
        return None
    
    counts = df[target_col].value_counts()
    fig = go.Figure(data=[
        go.Bar(x=['No Default', 'Default'], 
               y=[counts.get(0, 0), counts.get(1, 0)],
               marker_color=['#00CC96', '#EF553B'])
    ])
    fig.update_layout(
        title='Default Distribution',
        xaxis_title='Class',
        yaxis_title='Count',
        template='plotly_white',
        height=400
    )
    return fig

def plot_feature_importance(model, top_n=20):
    """Plot feature importance from model."""
    try:
        # Get final estimator
        if hasattr(model, 'named_steps'):
            final_estimator = list(model.named_steps.values())[-1]
        else:
            final_estimator = model
        
        if not hasattr(final_estimator, 'feature_importances_'):
            return None
        
        importances = final_estimator.feature_importances_
        
        # Get feature names
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        fig = px.bar(importance_df, x='importance', y='feature',
                     orientation='h',
                     title=f'Top {top_n} Feature Importances',
                     color='importance',
                     color_continuous_scale='Viridis')
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white',
            height=600
        )
        return fig
    except Exception as e:
        st.warning(f"Could not plot feature importance: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Default', 'Predicted Default'],
        y=['Actual No Default', 'Actual Default'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20}
    ))
    fig.update_layout(
        title='Confusion Matrix',
        template='plotly_white',
        height=400
    )
    return fig

def plot_roc_curve(y_true, y_proba):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        mode='lines',
        line=dict(color='#636EFA', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_white',
        height=500
    )
    return fig

def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_proba)
        except:
            pass
    
    return metrics

def plot_gauge(probability, title="Default Probability"):
    """Create a gauge chart for probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#00CC96'},
                {'range': [30, 60], 'color': '#FFA15A'},
                {'range': [60, 100], 'color': '#EF553B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def get_data_summary(df):
    """Get summary statistics for dataframe."""
    summary = {
        'Rows': len(df),
        'Columns': len(df.columns),
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    return summary
