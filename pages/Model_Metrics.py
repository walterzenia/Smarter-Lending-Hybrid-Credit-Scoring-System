"""
Model Metrics Page - Display Training Metrics from Pickle Files
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from apps.utils import (
    load_model, get_available_models, get_model_type, load_data, get_predictions, compute_metrics
)

st.set_page_config(page_title="Model Metrics", page_icon="", layout="wide")


def evaluate_ensemble_on_test_data(model, model_type):
    """Evaluate ensemble model on test data"""
    
    st.markdown("---")
    st.subheader(" Evaluation on Training Dataset")
    
    # For ensemble: use smoke_hybrid_features.csv (the training data)
    if model_type == 'ensemble':
        # Load the wrapper model instead of raw booster
        wrapper_path = "models/model_ensemble_wrapper.pkl"
        if Path(wrapper_path).exists():
            import sys
            import joblib
            # Add src to path so ensemble_model can be imported
            sys.path.insert(0, 'src')
            model = joblib.load(wrapper_path)
        else:
            st.error(f" Ensemble wrapper not found: {wrapper_path}")
            st.info("The ensemble needs the wrapper to generate meta-features from base models")
            return
        
        test_file = "data/smoke_hybrid_features.csv"
     
    elif model_type == 'behavioral':
        test_file = "data/test_behavioral_high_risk.csv"
    elif model_type == 'traditional':
        test_file = "data/test_traditional_high_risk.csv"
    else:
        st.error("Unknown model type")
        return
    
    if not Path(test_file).exists():
        st.error(f"Test data not found: {test_file}")
        return
    
    st.info(f" Loading data from: `{test_file}`")
    df_test = load_data(test_file)
    
    if df_test is None:
        st.error("Failed to load test data")
        return
    
    st.success(f"Loaded {len(df_test)} test samples")
    
    # Determine target column
    if 'TARGET' in df_test.columns:
        target_col = 'TARGET'
    elif 'target' in df_test.columns:
        target_col = 'target'
    elif 'default.payment.next.month' in df_test.columns:
        target_col = 'default.payment.next.month'
    else:
        st.error("No target column found in test data")
        return
    
    # Separate features and target
    X_test = df_test.drop(target_col, axis=1)
    y_test = df_test[target_col].values
    
    # Remove NaN values from target
    valid_mask = ~pd.isna(y_test)
    if not valid_mask.all():
        st.warning(f" Removing {(~valid_mask).sum()} rows with NaN target values")
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask]
    
    st.info(f"Features: {X_test.shape[1]} | Samples: {len(y_test)} | Defaults: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")
    
    # Get predictions
    with st.spinner("Generating predictions..."):
        y_pred, y_proba = get_predictions(model, X_test)
    
    if y_pred is None:
        st.error(" Prediction failed")
        return
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    # Display metrics
    st.markdown("###  Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['Precision']:.4f}")
    col3.metric("Recall", f"{metrics['Recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    
    if 'AUC-ROC' in metrics:
        col5.metric("AUC-ROC", f"{metrics['AUC-ROC']:.4f}")
    
    # Confusion matrix
    st.markdown("---")
    st.markdown("###  Prediction Distribution")
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction distribution
        pred_dist = pd.DataFrame({
            'Prediction': ['No Default', 'Default'],
            'Count': [(y_pred == 0).sum(), (y_pred == 1).sum()]
        })
        
        fig2 = go.Figure(data=[
            go.Bar(x=pred_dist['Prediction'], y=pred_dist['Count'],
                   text=pred_dist['Count'], textposition='auto')
        ])
        
        fig2.update_layout(
            title="Prediction Distribution",
            xaxis_title="Prediction",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # ROC Curve
    st.markdown("---")
    st.markdown("###  ROC Curve")
    
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='blue', width=2)
    ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_roc.update_layout(
        title=f"ROC Curve (AUC = {roc_auc:.4f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Feature Importance for Ensemble
    st.markdown("---")
    st.markdown("###  Feature Importance")
    
    # Get the meta-model from wrapper
    if hasattr(model, 'meta_model'):
        meta_model = model.meta_model
        
        # LightGBM Booster has feature_importance method
        if hasattr(meta_model, 'feature_importance'):
            importance_scores = meta_model.feature_importance(importance_type='gain')
            feature_names = meta_model.feature_name()
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=False)
            
            # Plot all features
            fig_imp = go.Figure()
            
            fig_imp.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['Importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig_imp.update_layout(
                title="Meta-Learner Feature Importance",
                xaxis_title="Importance Score (Gain)",
                yaxis_title="Meta-Feature",
                height=max(400, len(feature_names) * 25),
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Show features table
            with st.expander("📋View All Meta-Features"):
                st.dataframe(
                    importance_df.reset_index(drop=True),
                    width='stretch'
                )
                
                st.info("""
                **Meta-Features Explained:**
                - `pred_traditional`, `pred_behavioral`: Base model predictions
                - `pred_avg`, `pred_max`, `pred_min`: Aggregated predictions
                - `pred_diff`, `pred_ratio`: Prediction differences
                - `trad_feature_*`: Top 10 traditional model features
                - `behav_feature_*`: Top 10 behavioral model features
                """)
    
    # Summary
    st.markdown("---")
    st.success(f"""
    ** Evaluation Complete**
    
    - Model correctly classified **{metrics['Accuracy']*100:.2f}%** of test samples
    - Of predicted defaults, **{metrics['Precision']*100:.2f}%** were actual defaults (Precision)
    - Model caught **{metrics['Recall']*100:.2f}%** of all defaults (Recall)
    - AUC-ROC score: **{metrics.get('AUC-ROC', 0):.4f}**
    """)

def display_stored_metrics(model, model_name, model_type):
    """Display metrics stored in the model from training"""
    
    st.markdown("---")
    st.subheader(" Training Metrics")
    
    try:
        # Check if model has stored metrics
        if not hasattr(model, 'best_score_'):
            st.warning(" This model doesn't have stored training metrics stored in sklearn format")
            st.info(" Evaluating on test data instead...")
            evaluate_ensemble_on_test_data(model, model_type)
            return
            
        best_score = model.best_score_
        
        # Display final metrics
        st.markdown("### Final Validation Metrics")
        
        metric_cols = st.columns(len(best_score))
        
        for idx, (dataset_name, metrics) in enumerate(best_score.items()):
            with metric_cols[idx]:
                st.markdown(f"**{dataset_name.replace('_', ' ').title()}**")
                for metric_name, value in metrics.items():
                    st.metric(
                        label=metric_name.upper(),
                        value=f"{value:.4f}"
                    )
        
        # Plot training curves if available
        if hasattr(model, 'evals_result_'):
            st.markdown("---")
            st.markdown("### Training History")
            
            evals_result = model.evals_result_
            
            # Create tabs for different datasets
            dataset_names = list(evals_result.keys())
            tabs = st.tabs(dataset_names)
            
            for tab, dataset_name in zip(tabs, dataset_names):
                with tab:
                    metrics_data = evals_result[dataset_name]
                    
                    # Plot each metric
                    for metric_name, values in metrics_data.items():
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(len(values))),
                            y=values,
                            mode='lines',
                            name=metric_name.upper(),
                            line=dict(width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"{metric_name.upper()} over Iterations ({dataset_name})",
                            xaxis_title="Iteration",
                            yaxis_title=metric_name.upper(),
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        # Display best iteration info
        if hasattr(model, 'best_iteration_'):
            st.markdown("---")
            st.markdown("### Model Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Best Iteration",
                    value=model.best_iteration_
                )
            
            with col2:
                if hasattr(model, 'n_features_in_'):
                    st.metric(
                        label="Number of Features",
                        value=model.n_features_in_
                    )
            
            with col3:
                if hasattr(model, 'n_estimators'):
                    st.metric(
                        label="Total Estimators",
                        value=model.n_estimators
                    )
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.markdown("### Feature Importance")
            
            feature_importance = model.feature_importances_
            
            # Get feature names if available
            if hasattr(model, 'feature_name_'):
                feature_names = model.feature_name_
            elif hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(20)
            
            # Plot top 20 features
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['Importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title="Top 20 Most Important Features",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=600,
                yaxis={'categoryorder':'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features table
            with st.expander(" View Top Features Table"):
                st.dataframe(
                    importance_df.reset_index(drop=True),
                    width='stretch'
                )
        
    except Exception as e:
        st.error(f" Error displaying metrics: {str(e)}")
        st.exception(e)

def show():
    st.title(" Model Performance Metrics")
    st.markdown("View training metrics and performance stored in model files")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader(" Select Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        models = get_available_models()
        
        if not models:
            st.error(" No models found in models/ directory")
            return
        
        model_names = [Path(m).name for m in models]
        selected_model_name = st.selectbox("Select Model", model_names, key="model_select")
        selected_model_path = models[model_names.index(selected_model_name)]
    
    with col2:
        # Get model type
        model_type = get_model_type(selected_model_name)
        
        st.info(f"**Selected:** {selected_model_name}")
        
        # Model type indicators
        if model_type == 'ensemble':
            st.markdown(" **Ensemble Hybrid Model**")
        elif model_type == 'traditional':
            st.markdown(" **Traditional Model**")
        elif model_type == 'behavioral':
            st.markdown(" **Behavioral Model**")
    
    # Load model
    with st.spinner("Loading model..."):
        # For ensemble, use the wrapper instead of raw booster
        if model_type == 'ensemble' and 'ensemble' in selected_model_name.lower():
            wrapper_path = "models/model_ensemble_wrapper.pkl"
            if Path(wrapper_path).exists():
                import joblib
                model = joblib.load(wrapper_path)
                st.info("ℹ Loaded ensemble wrapper (handles meta-feature generation)")
            else:
                model = load_model(selected_model_path)
                st.warning(" Using raw booster (wrapper not found)")
        else:
            model = load_model(selected_model_path)
    
    if model is None:
        st.error(" Failed to load model")
        return
    
    st.success(" Model loaded successfully")
    
    # Display stored metrics
    display_stored_metrics(model, selected_model_name, model_type)
    
    # Info box
    st.markdown("---")
    
    # Check model type to show appropriate info
    if model_type == 'ensemble' and not hasattr(model, 'best_score_'):
        st.info("""
        **ℹ About These Metrics - Ensemble Meta-Learner**
        
        ** Meta-Learner Architecture:**
        1. **Base Models**: 
           - Traditional (Home Credit features) → Prediction A
           - Behavioral (UCI Credit Card features) → Prediction B
        2. **Meta-Features**: Creates 27 features from:
           - Both model predictions (pred_A, pred_B, avg, max, min, diff, ratio)
           - Top 10 features from each base model
        3. **Final Prediction**: LightGBM meta-model combines everything
        
        ** Evaluation Data:**
        - **Dataset**: `smoke_hybrid_features.csv` (1,000 samples from smoke_engineered.csv)
        - **Contains**: Engineered Home Credit features
        - **Target**: `TARGET` (loan default)
        - **Note**: This is the training data - showing in-sample performance
        
        ** Why This Approach:**
        - Combines strengths of two specialized models
        - Learns optimal weighting of predictions
        - Meta-learner corrects individual model biases
        """)
    else:
        st.info("""
        **ℹ About These Metrics**
        
        These metrics were computed during model training and stored in the pickle file:
        - **Training Metrics**: Performance on the training dataset
        - **Validation Metrics**: Performance on the validation dataset (if available)
        - **Best Iteration**: Optimal number of boosting rounds (early stopping)
        - **Feature Importance**: Relative importance of features in making predictions
        
        These metrics reflect the model's performance during training and do not involve any new data evaluation.
        """)

if __name__ == "__main__":
    show()
else:
    show()

