"""
Loan Default Prediction Dashboard
Main Application Entry Point
"""
import warnings
import streamlit as st

# Suppress all warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 100%;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 2rem; color: #1f77b4;'></h1>
        <h2 style='font-size: 1.3rem; margin-top: 0;'>Loan Default<br>Prediction</h2>
        <p style='color: #666; font-size: 0.9rem;'>AI-Powered Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("###  Navigation")
    st.info("""
    Use the sidebar to navigate between pages:
    - **Home**: Dashboard overview
    - **EDA**: Explore data
    - **Prediction**: Make predictions
    - **Feature Importance**: Model insights
    - **Model Metrics**: Performance evaluation
    """)
    
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    This dashboard uses **LightGBM** models trained on loan application data to predict default risk.
    
    **Features:**
    - Real-time predictions
    - Batch processing
    - SHAP explanations
    - Performance tracking
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: #999;'>
        <p>© 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h1 style='font-size: 3rem; color: #1f77b4;'> Loan Default Prediction Dashboard</h1>
    <p style='font-size: 1.2rem; color: #555;'>AI-Powered Risk Assessment for Financial Lending</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ###  Welcome to the Loan Default Prediction System
    
    This advanced machine learning platform helps financial institutions assess loan default risk 
    using state-of-the-art LightGBM models trained on comprehensive applicant data.

    #### What This App Does:

    - **Explore Data**: Analyze applicant demographics, financial history, and behavioral patterns
    - **Predict Defaults**: Generate real-time predictions for loan default likelihood
    - **Explain Models**: Understand which features drive predictions using SHAP values
    - **Monitor Performance**: Track model accuracy, AUC, and other key metrics

    ####  Key Features:
    
    - **Batch Predictions**: Upload datasets for bulk risk assessment
    - **Single Applicant Analysis**: Manual input for individual loan evaluation
    - **Interactive Visualizations**: Explore data patterns with dynamic charts
    - **Model Interpretability**: SHAP-based explanations for transparency
    - ** Ensemble Hybrid Model**: Best performance combining Traditional + Behavioral features
    """)

with col2:
    st.markdown("""
    ###  Quick Start Guide
    
    **Step 1:** Navigate using the sidebar
    
    **Step 2:** Start with EDA to explore data
    
    **Step 3:** Make predictions on new applicants
    
    **Step 4:** Review feature importance
    
    **Step 5:** Check model performance
    """)
    
    st.info(" **Tip**: Use the sidebar to navigate between pages")

st.markdown("---")

# App Capabilities
st.markdown("###  Dashboard Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='padding: 1.5rem; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
        <h3 style='color: #1f77b4;'> EDA</h3>
        <p style='color: #555;'>Comprehensive data exploration with summary statistics, distributions, and correlations</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='padding: 1.5rem; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
        <h3 style='color: #ff7f0e;'> Predictions</h3>
        <p style='color: #555;'>Real-time loan default probability with risk classification and confidence scores</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='padding: 1.5rem; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
        <h3 style='color: #2ca02c;'> Insights</h3>
        <p style='color: #555;'>Feature importance and SHAP values explain model decisions transparently</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model Information
st.markdown("### Model Architecture")

st.markdown("""
This system employs **LightGBM** (Light Gradient Boosting Machine), an efficient gradient boosting framework 
optimized for high performance and accuracy.

**Available Models:**
- **Traditional Features**: Credit history, income, demographics (AUC: ~0.75)
- **Behavioral Features**: Payment patterns, spending behavior (AUC: ~0.76)
- ** Ensemble Hybrid**: Combined traditional + behavioral **(AUC: 0.8591 - Best Performance!)**
""")

st.markdown("---")

# Navigation Buttons
st.markdown("###  Quick Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button(" Explore Data", use_container_width=True):
        st.switch_page("pages/EDA.py")

with col2:
    if st.button(" Make Prediction", use_container_width=True):
        st.switch_page("pages/Prediction.py")

with col3:
    if st.button(" Feature Insights", use_container_width=True):
        st.switch_page("pages/Feature_Importance.py")

with col4:
    if st.button(" View Metrics", use_container_width=True):
        st.switch_page("pages/Model_Metrics.py")

st.markdown("---")

# Footer
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <p style='color: #999; font-size: 0.9rem;'>
            © 2025 Loan Default Prediction System
        </p>
    </div>
    """, unsafe_allow_html=True)
