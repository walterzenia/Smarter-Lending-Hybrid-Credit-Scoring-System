"""
Feature Importance Page - Model Interpretability
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from apps.utils import (
    load_model, get_available_models, plot_feature_importance,
    get_predictions
)

st.set_page_config(page_title="Feature Importance", page_icon="", layout="wide")

def show():
    st.title("Feature Importance & Model Interpretability")
    st.markdown("Understand which features drive model predictions using SHAP values and feature importance")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("Select Model")
    
    models = get_available_models()
    
    if not models:
        st.error("No models found. Please train a model first.")
        return
    
    model_names = [Path(m).name for m in models]
    selected_model_name = st.selectbox("Choose Model", model_names)
    selected_model_path = models[model_names.index(selected_model_name)]
    
    model = load_model(selected_model_path)
    
    if model is None:
        st.error("Failed to load model")
        return

    st.success(f" Model loaded: {selected_model_name}")

    st.markdown("---")
    
    # Tabs for different interpretability views
    tab1, tab2, tab3 = st.tabs(["Global Feature Importance", " SHAP Analysis", " Feature Details"])
    
    with tab1:
        global_importance(model)
    
    with tab2:
        shap_analysis(model)
    
    with tab3:
        feature_details(model)

def global_importance(model):
    """Display global feature importance"""
    st.markdown("### Global Feature Importance")
    st.markdown("Shows which features have the most impact on predictions across all samples")
    
    top_n = st.slider("Number of top features to display", 10, 50, 20)
    
    fig = plot_feature_importance(model, top_n=top_n)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Interpretation Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **What is Feature Importance?**
            
            Feature importance measures how much each feature contributes to the model's predictions.
            
            - **Higher values** = More influential features
            - **Lower values** = Less influential features
            
            The model uses these features to split decision trees.
            """)
        
        with col2:
            st.success("""
            **How to Use This Information:**
            
            - Focus data quality efforts on top features
            - Investigate why certain features rank high/low
            - Consider feature engineering for high-impact variables
            - Remove low-importance features to simplify model
            """)
    else:
        st.warning("Could not generate feature importance plot. Model may not support feature_importances_.")

def shap_analysis(model):
    """SHAP-based model interpretability"""
    st.markdown("### SHAP (SHapley Additive exPlanations)")
    st.markdown("SHAP values explain individual predictions by showing each feature's contribution")
    
    st.info("""
     **About SHAP:**
    
    SHAP provides both global and local interpretability:
    - **Global**: Overall feature impact across dataset
    - **Local**: How features affect a single prediction
    """)
    
    analysis_type = st.radio(
        "Choose analysis type:",
        ["Global SHAP Summary", "Local Explanation (Single Prediction)"],
        horizontal=True
    )
    
    if analysis_type == "Global SHAP Summary":
        global_shap(model)
    else:
        local_shap(model)

def global_shap(model):
    """Global SHAP summary plot"""
    st.markdown("#### Global SHAP Summary")
    
    st.info("""
    ℹ **Computing SHAP values requires:**
    - Sample dataset for analysis
    - SHAP library installed
    - May take several minutes for large datasets
    """)
    
    # Data upload for SHAP
    uploaded_file = st.file_uploader("Upload sample data (CSV) for SHAP analysis", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Sample for performance
        max_samples = st.slider("Max samples for SHAP computation", 100, 1000, 500)
        
        if len(df) > max_samples:
            df_sample = df.sample(max_samples, random_state=42)
            st.info(f"Sampled {max_samples} rows from {len(df)} for SHAP analysis")
        else:
            df_sample = df
        
        if st.button(" Compute SHAP Values", type="primary"):
            try:
                import shap
                
                with st.spinner("Computing SHAP values... This may take a few minutes"):
                    # Get final estimator
                    if hasattr(model, 'named_steps'):
                        final_estimator = list(model.named_steps.values())[-1]
                    else:
                        final_estimator = model
                    
                    # Prepare data
                    X = df_sample.copy()
                    if 'TARGET' in X.columns:
                        X = X.drop('TARGET', axis=1)
                    
                    # Align features
                    from apps.utils import align_features
                    X_aligned = align_features(X, model)
                    
                    # Compute SHAP
                    explainer = shap.TreeExplainer(final_estimator)
                    shap_values = explainer.shap_values(X_aligned)
                    
                    # Plot
                    st.success(" SHAP values computed successfully!")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_aligned, plot_type="bar", show=False)
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    
                    # Detailed summary
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_aligned, show=False)
                    st.pyplot(fig2)
                    
                    st.markdown("""
                    **How to read this plot:**
                    - Each dot is a sample
                    - Red = High feature value
                    - Blue = Low feature value
                    - X-axis = SHAP value (impact on prediction)
                    - Positive SHAP = Increases default probability
                    - Negative SHAP = Decreases default probability
                    """)
                    
            except ImportError:
                st.error("SHAP library not installed. Install with: pip install shap")
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
    else:
        st.warning("Please upload a CSV file to compute SHAP values")

def local_shap(model):
    """Local SHAP explanation for single prediction"""
    st.markdown("#### Local Explanation - Single Applicant")
    st.markdown("Upload a single-row CSV or enter data manually to see how each feature contributes to the prediction")
    
    input_method = st.radio("Input method:", ["Upload CSV", "Manual Entry"], horizontal=True)
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload single-row CSV", type=["csv"], key="local_shap")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            if len(df) > 1:
                st.warning(f"File contains {len(df)} rows. Using only the first row.")
                df = df.head(1)
            
            st.dataframe(df, use_container_width=True)
            
            if st.button("🔍 Generate SHAP Explanation", type="primary"):
                generate_local_shap(model, df)
    else:
        st.info("Manual entry for local SHAP - Use the Prediction page for detailed manual input, then return here with results")

def generate_local_shap(model, X):
    """Generate local SHAP explanation"""
    try:
        import shap
        
        with st.spinner("Computing SHAP explanation..."):
            # Get final estimator
            if hasattr(model, 'named_steps'):
                final_estimator = list(model.named_steps.values())[-1]
            else:
                final_estimator = model
            
            # Align features
            from apps.utils import align_features
            X_aligned = align_features(X, model)
            
            # Prediction
            pred, prob = get_predictions(model, X)
            
            if pred is not None:
                st.markdown("### Prediction")
                col1, col2 = st.columns(2)
                col1.metric("Prediction", "Default" if pred[0] == 1 else "No Default")
                if prob is not None:
                    col2.metric("Probability", f"{prob[0] * 100:.2f}%")
            
            # SHAP
            explainer = shap.TreeExplainer(final_estimator)
            shap_values = explainer.shap_values(X_aligned)
            
            st.markdown("### Feature Contributions")
            
            # Waterfall plot
            st.markdown("#### Waterfall Plot")
            fig = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_aligned.iloc[0],
                feature_names=X_aligned.columns.tolist()
            ), show=False)
            st.pyplot(fig)
            
            st.markdown("""
            **How to read:**
            - Starting point (E[f(x)]): Model's base prediction
            - Red bars: Features pushing prediction towards default
            - Blue bars: Features pushing prediction away from default
            - Final value (f(x)): Actual prediction for this applicant
            """)
            
            st.success("SHAP explanation generated successfully!")
            
    except ImportError:
        st.error("SHAP library not installed. Install with: pip install shap")
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

def feature_details(model):
    """Display feature details and statistics"""
    st.markdown("### Feature Details")
    
    try:
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
            
            st.write(f"**Total Features:** {len(features)}")
            
            # Feature list
            features_df = pd.DataFrame({
                'Feature Name': features,
                'Index': range(len(features))
            })
            
            st.dataframe(features_df, use_container_width=True)
            
            # Search
            search = st.text_input("Search features:")
            if search:
                filtered = features_df[features_df['Feature Name'].str.contains(search, case=False)]
                st.dataframe(filtered, use_container_width=True)
        else:
            st.warning("Feature names not available in this model")
    except Exception as e:
        st.error(f"Could not retrieve feature details: {e}")

if __name__ == "__main__":
    show()
else:
    show()
