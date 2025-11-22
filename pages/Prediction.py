"""
Prediction Page - Loan Default Risk Assessment
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from apps.utils import (
    load_model, get_available_models, get_predictions,
    classify_risk, plot_gauge, get_model_type
)

st.set_page_config(page_title="Prediction - Loan Default", page_icon="", layout="wide")

def show():
    st.title("Loan Default Prediction")
    st.markdown("Predict loan default risk using trained LightGBM models")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader(" Model Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        models = get_available_models()
        
        if not models:
            st.error("No models found in the models/ directory. Please train and save a model first.")
            return
        
        model_names = [Path(m).name for m in models]
        selected_model_name = st.selectbox(
            "Select Model",
            model_names,
            help="Choose from Traditional, Behavioral, or Hybrid feature models"
        )
        
        selected_model_path = models[model_names.index(selected_model_name)]
    
    with col2:
        # Get model type
        model_type = get_model_type(selected_model_name)
        
        st.info(f"**Selected:** {selected_model_name}")
        
        # Model type indicators
        if model_type == 'ensemble':
            st.markdown(" **Ensemble Hybrid Model**")
            st.caption("Combines Traditional + Behavioral features")
        elif model_type == 'traditional':
            st.markdown("**Traditional Model**")
            st.caption("Home Credit credit history data")
        elif model_type == 'behavioral':
            st.markdown(" **Behavioral Model**")
            st.caption("UCI credit card payment patterns")
    
    # Load model
    model = load_model(selected_model_path)
    
    if model is None:
        st.error("Failed to load model")
        return
    
    st.success(f" Model loaded successfully")
    
    st.markdown("---")
    
    # Prediction Mode
    st.subheader(" Prediction Mode")
    
    pred_mode = st.radio(
        "Choose prediction mode:",
        ["Batch Prediction (Upload CSV)", "Single Applicant (Manual Input)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if pred_mode == "Batch Prediction (Upload CSV)":
        batch_prediction(model, selected_model_name)
    else:
        manual_prediction(model, selected_model_name)

def batch_prediction(model, model_name):
    """Handle batch predictions from uploaded CSV"""
    st.markdown("###  Batch Prediction")
    st.markdown("Upload a CSV file with applicant data for bulk risk assessment")
    
    # Determine model type
    model_type = get_model_type(model_name)
    
    if model_type == 'ensemble':
        st.info(" **Ensemble Model:** Requires hybrid features (traditional + behavioral). Use `smoke_hybrid_features.csv`.")
    elif model_type == 'traditional':
        st.info(" **Traditional Model:** Requires Home Credit features. Use `smoke_engineered.csv`.")
    elif model_type == 'behavioral':
        st.info(" **Behavioral Model:** Requires UCI credit card features. Use `uci_interface_test.csv`.")
    
    uploaded_file = st.file_uploader(" Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.write(f"**Loaded:** {len(df)} applicants")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button(" Run Predictions", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                predictions, probabilities = get_predictions(model, df)
                
                if predictions is None:
                    st.error(" Prediction failed. Please check your data format.")
                    if model_type == 'ensemble':
                        st.warning(" Ensemble model requires both traditional and behavioral features. Upload a hybrid dataset.")
                    elif model_type == 'traditional':
                        st.warning(" Traditional model requires Home Credit features (EXT_SOURCE, AMT_CREDIT, etc.).")
                    elif model_type == 'behavioral':
                        st.warning(" Behavioral model requires UCI features (PAY_*, BILL_AMT*, PAY_AMT*, etc.).")
                    return
                
                # Create results dataframe
                results = df.copy()
                results['Prediction'] = predictions
                
                if probabilities is not None:
                    results['Default Probability'] = probabilities
                    results['Risk Level'] = [classify_risk(p) for p in probabilities]
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total = len(predictions)
                defaults = predictions.sum()
                no_defaults = total - defaults
                
                col1.metric("Total Applicants", total)
                col2.metric("Predicted Defaults ", defaults)
                col3.metric("No Defaults ", no_defaults)
                
                if probabilities is not None:
                    avg_prob = probabilities.mean() * 100
                    col4.metric("Avg Default Prob", f"{avg_prob:.1f}%")
                
                # Results table
                st.markdown("####  Detailed Results")
                
                # Add ID column if not present
                if 'SK_ID_CURR' not in results.columns and 'ID' not in results.columns:
                    results.insert(0, 'ID', range(1, len(results) + 1))
                
                # Show key columns
                display_cols = ['SK_ID_CURR'] if 'SK_ID_CURR' in results.columns else ['ID']
                display_cols += ['Prediction']
                if 'Default Probability' in results.columns:
                    display_cols += ['Default Probability', 'Risk Level']
                
                st.dataframe(
                    results[display_cols].style.format({
                        'Default Probability': '{:.2%}' if 'Default Probability' in display_cols else None
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Risk distribution
                if 'Risk Level' in results.columns:
                    st.markdown("####  Risk Distribution")
                    risk_counts = results['Risk Level'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric(" Low Risk", risk_counts.get("Low Risk ", 0))
                    col2.metric(" Medium Risk", risk_counts.get("Medium Risk ", 0))
                    col3.metric(" High Risk", risk_counts.get("High Risk ", 0))
                
                # Download results
                st.markdown("---")
                csv = results.to_csv(index=False)
                st.download_button(
                    "⬇ Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

def manual_prediction(model, model_name):
    """Handle manual single applicant prediction"""
    st.markdown("###  Single Applicant Prediction")
    st.markdown("Enter applicant details manually for individual risk assessment")
    
    # Determine model type
    model_type = get_model_type(model_name)
    
    st.info(f" Fill in the applicant information for **{model_type.upper()}** model. Default values are provided for demonstration.")
    
    # Create model-specific input form
    if model_type == 'traditional':
        input_data, submitted = traditional_input_form()
    elif model_type == 'behavioral':
        input_data, submitted = behavioral_input_form()
    elif model_type == 'ensemble':
        input_data, submitted = hybrid_input_form()
    else:
        st.error("Unknown model type")
        return
    
    if submitted:
        with st.spinner("Analyzing applicant..."):
            print(f"\n[PREDICTION] Making prediction with {model_name}")
            print(f"[PREDICTION] Input data shape: {input_data.shape}")
            print(f"[PREDICTION] Input features: {list(input_data.columns)[:10]}...")
            
            predictions, probabilities = get_predictions(model, input_data)
            
            if predictions is None:
                st.error(" Prediction failed. The model may require additional features.")
                if model_type == 'ensemble':
                    st.info(" Ensemble model requires both traditional and behavioral features. Upload a CSV file with complete hybrid data.")
                elif model_type == 'traditional':
                    st.info(" Traditional model works best with Home Credit features. For full accuracy, upload `smoke_engineered.csv`.")
                elif model_type == 'behavioral':
                    st.info(" Behavioral model requires UCI credit card features. For full accuracy, upload `uci_interface_test.csv`.")
                return
            
            display_prediction_results(predictions, probabilities, model_type)

def traditional_input_form():
    """Form for Traditional (Home Credit) features only"""
    with st.form("traditional_form"):
        st.markdown("####  Traditional Model Features")
        st.warning("""
        ⚠️ **Important Limitation:**  
        The traditional model was trained on **487 features** including bureau reports, previous applications, 
        installments, POS-CASH, and credit card data. This manual form only provides **11 basic features** (→24 after engineering).
        
        **Result:** Predictions may fail or be unreliable due to missing features.  
        **Recommended:** Use **Batch Prediction** with `smoke_engineered.csv` for accurate traditional model predictions.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Credit Bureau Scores**")
            ext_source_1 = st.slider("External Source 1", 0.0, 1.0, 0.5, 0.01, help="Credit bureau score 1")
            ext_source_2 = st.slider("External Source 2", 0.0, 1.0, 0.5, 0.01, help="Credit bureau score 2")
            ext_source_3 = st.slider("External Source 3", 0.0, 1.0, 0.5, 0.01, help="Credit bureau score 3")
        
        with col2:
            st.markdown("**Loan Details**")
            credit_amount = st.number_input("Credit Amount", 10000, 1000000, 200000, step=10000)
            annuity = st.number_input("Loan Annuity (monthly)", 1000, 50000, 15000, step=1000)
            goods_price = st.number_input("Goods Price", 10000, 1000000, 180000, step=10000)
        
        with col3:
            st.markdown("**Personal Details**")
            age = st.number_input("Age (years)", 18, 80, 35)
            income = st.number_input("Annual Income", 20000, 500000, 150000, step=10000)
            employment_years = st.slider("Years Employed", 0, 40, 5)
            fam_members = st.number_input("Family Members", 1, 10, 2)
        
        submitted = st.form_submit_button(" Predict Default Risk", type="primary", use_container_width=True)
    
    if submitted:
        days_birth = age * -365
        days_employed = employment_years * -365
        credit_income_ratio = credit_amount / income if income > 0 else 0
        
        input_data = pd.DataFrame({
            'EXT_SOURCE_1': [ext_source_1],
            'EXT_SOURCE_2': [ext_source_2],
            'EXT_SOURCE_3': [ext_source_3],
            'AMT_CREDIT': [credit_amount],
            'AMT_INCOME_TOTAL': [income],
            'AMT_ANNUITY': [annuity],
            'AMT_GOODS_PRICE': [goods_price],
            'DAYS_BIRTH': [days_birth],
            'DAYS_EMPLOYED': [days_employed],
            'CNT_FAM_MEMBERS': [fam_members],
            'OWN_CAR_AGE': [np.nan],  # Not collected in form
        })
        
        # Apply traditional feature engineering
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from feature_engineering import process_apps
        
        print(f"\n[TRADITIONAL] Before feature engineering: {len(input_data.columns)} features")
        input_data = process_apps(input_data)
        print(f"[TRADITIONAL] After feature engineering: {len(input_data.columns)} features")
        print(f"[TRADITIONAL] First 10 features: {list(input_data.columns)[:10]}")
        
        return input_data, True
    
    return None, False

def behavioral_input_form():
    """Form for Behavioral (UCI Credit Card) features only"""
    with st.form("behavioral_form"):
        st.markdown("####  Behavioral Model Features")
        st.info("This form provides UCI Credit Card features. For all 31 features, use batch prediction.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Personal Information**")
            limit_bal = st.number_input("Credit Limit", 10000, 500000, 100000, step=10000, help="Credit card limit")
            age = st.number_input("Age", 21, 70, 35)
            sex = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
            marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"])
        
        with col2:
            st.markdown("**Payment History (6 Months)**")
            st.caption("-1=On time, 0=Revolving, 1-9=Months delayed")
            pay_0 = st.selectbox("Current Month Status", [-1, 0, 1, 2, 3], index=0)
            pay_2 = st.selectbox("2 Months Ago", [-1, 0, 1, 2, 3], index=0)
            pay_3 = st.selectbox("3 Months Ago", [-1, 0, 1, 2, 3], index=0)
            pay_4 = st.selectbox("4 Months Ago", [-1, 0, 1, 2, 3], index=0)
            pay_5 = st.selectbox("5 Months Ago", [-1, 0, 1, 2, 3], index=0)
            pay_6 = st.selectbox("6 Months Ago", [-1, 0, 1, 2, 3], index=0)
        
        with col3:
            st.markdown("**Bill Amounts (6 Months)**")
            bill_amt1 = st.number_input("Last Month Bill", 0, 500000, 50000, step=5000)
            bill_amt2 = st.number_input("2 Months Ago Bill", 0, 500000, 48000, step=5000)
            bill_amt3 = st.number_input("3 Months Ago Bill", 0, 500000, 46000, step=5000)
            
            st.markdown("**Payment Amounts (6 Months)**")
            pay_amt1 = st.number_input("Last Month Payment", 0, 200000, 20000, step=2000)
            pay_amt2 = st.number_input("2 Months Ago Payment", 0, 200000, 19000, step=2000)
            pay_amt3 = st.number_input("3 Months Ago Payment", 0, 200000, 18000, step=2000)
        
        # Hidden fields for remaining months (simplified)
        bill_amt4 = bill_amt3 * 0.95
        bill_amt5 = bill_amt3 * 0.90
        bill_amt6 = bill_amt3 * 0.85
        pay_amt4 = pay_amt3 * 0.95
        pay_amt5 = pay_amt3 * 0.90
        pay_amt6 = pay_amt3 * 0.85

        submitted = st.form_submit_button("🔮 Predict Default Risk", type="primary", use_container_width=True)

    if submitted:
        # Map categorical values
        sex_val = 1 if sex == "Male" else 2
        edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
        edu_val = edu_map[education]
        mar_map = {"Married": 1, "Single": 2, "Others": 3}
        mar_val = mar_map[marriage]
        
        # Create dataframe with UCI base features (need BILL_AMT and PAY_AMT for feature engineering)
        input_data = pd.DataFrame({
            'LIMIT_BAL': [limit_bal],
            'SEX': [sex_val],
            'EDUCATION': [edu_val],
            'MARRIAGE': [mar_val],
            'AGE': [age],
            'PAY_0': [pay_0],
            'PAY_2': [pay_2],
            'PAY_3': [pay_3],
            'PAY_4': [pay_4],
            'PAY_5': [pay_5],
            'PAY_6': [pay_6],
            'BILL_AMT1': [bill_amt1],
            'BILL_AMT2': [bill_amt2],
            'BILL_AMT3': [bill_amt3],
            'BILL_AMT4': [bill_amt4],
            'BILL_AMT5': [bill_amt5],
            'BILL_AMT6': [bill_amt6],
            'PAY_AMT1': [pay_amt1],
            'PAY_AMT2': [pay_amt2],
            'PAY_AMT3': [pay_amt3],
            'PAY_AMT4': [pay_amt4],
            'PAY_AMT5': [pay_amt5],
            'PAY_AMT6': [pay_amt6],
        })
        
        # Apply behavioral feature engineering to create all 31 features
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from feature_engineering import behaviorial_features
        
        print(f"\n[BEHAVIORAL] Before feature engineering: {len(input_data.columns)} features")
        input_data = behaviorial_features(input_data)
        print(f"[BEHAVIORAL] After feature engineering: {len(input_data.columns)} features")
        print(f"[BEHAVIORAL] First 10 features: {list(input_data.columns)[:10]}")
        
        return input_data, True
    
    return None, False

def hybrid_input_form():
    """Form for Hybrid (Ensemble) features - combines traditional and behavioral"""
    with st.form("hybrid_form"):
        st.markdown("#### Ensemble Model Features")
        st.warning("This form combines Traditional + Behavioral features. For best results, use batch prediction with `smoke_hybrid_features.csv`.")
        
        st.markdown("##### Traditional Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Credit Bureau Scores**")
            ext_source_1 = st.slider("External Source 1", 0.0, 1.0, 0.5, 0.01, key="hyb_ext1")
            ext_source_2 = st.slider("External Source 2", 0.0, 1.0, 0.5, 0.01, key="hyb_ext2")
            ext_source_3 = st.slider("External Source 3", 0.0, 1.0, 0.5, 0.01, key="hyb_ext3")
        
        with col2:
            st.markdown("**Loan Details**")
            credit_amount = st.number_input("Credit Amount", 10000, 1000000, 200000, step=10000, key="hyb_credit")
            annuity = st.number_input("Loan Annuity", 1000, 50000, 15000, step=1000, key="hyb_annuity")
            goods_price = st.number_input("Goods Price", 10000, 1000000, 180000, step=10000, key="hyb_goods")
        
        with col3:
            st.markdown("**Personal Details**")
            age = st.number_input("Age (years)", 21, 70, 35, key="hyb_age")
            income = st.number_input("Annual Income", 20000, 500000, 150000, step=10000, key="hyb_income")
            employment_years = st.slider("Years Employed", 0, 40, 5, key="hyb_emp")
            fam_members = st.number_input("Family Members", 1, 10, 2, key="hyb_fam")
        
        st.markdown("---")
        st.markdown("##### Behavioral Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Credit Card Info**")
            limit_bal = st.number_input("Credit Card Limit", 10000, 500000, 100000, step=10000, key="hyb_limit")
            sex = st.selectbox("Gender", ["Male", "Female"], key="hyb_sex")
            education = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"], key="hyb_edu")
            marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"], key="hyb_mar")
        
        with col2:
            st.markdown("**Payment Status (6 Months)**")
            st.caption("-1=On time, 0=Revolving, 1-3=Delayed")
            pay_0 = st.selectbox("Current", [-1, 0, 1, 2, 3], index=0, key="hyb_pay0")
            pay_2 = st.selectbox("2 Months Ago", [-1, 0, 1, 2, 3], index=0, key="hyb_pay2")
            pay_3 = st.selectbox("3 Months Ago", [-1, 0, 1, 2, 3], index=0, key="hyb_pay3")
        
        with col3:
            st.markdown("**Bills & Payments**")
            bill_amt1 = st.number_input("Last Bill", 0, 200000, 50000, step=5000, key="hyb_bill")
            pay_amt1 = st.number_input("Last Payment", 0, 100000, 20000, step=2000, key="hyb_payamt")
        
        # Simplified remaining months
        pay_4 = pay_3
        pay_5 = pay_3
        pay_6 = pay_3
        bill_amt2 = bill_amt1 * 0.95
        bill_amt3 = bill_amt1 * 0.90
        bill_amt4 = bill_amt1 * 0.85
        bill_amt5 = bill_amt1 * 0.80
        bill_amt6 = bill_amt1 * 0.75
        pay_amt2 = pay_amt1 * 0.95
        pay_amt3 = pay_amt1 * 0.90
        pay_amt4 = pay_amt1 * 0.85
        pay_amt5 = pay_amt1 * 0.80
        pay_amt6 = pay_amt1 * 0.75
        
        submitted = st.form_submit_button(" Predict Default Risk", type="primary", use_container_width=True)
    
    if submitted:
        days_birth = age * -365
        days_employed = employment_years * -365
        credit_income_ratio = credit_amount / income if income > 0 else 0
        
        # Map categorical values
        sex_val = 1 if sex == "Male" else 2
        edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
        edu_val = edu_map[education]
        mar_map = {"Married": 1, "Single": 2, "Others": 3}
        mar_val = mar_map[marriage]
        
        # Combine both feature sets
        input_data = pd.DataFrame({
            # Traditional features
            'EXT_SOURCE_1': [ext_source_1],
            'EXT_SOURCE_2': [ext_source_2],
            'EXT_SOURCE_3': [ext_source_3],
            'AMT_CREDIT': [credit_amount],
            'AMT_INCOME_TOTAL': [income],
            'AMT_ANNUITY': [annuity],
            'AMT_GOODS_PRICE': [goods_price],
            'DAYS_BIRTH': [days_birth],
            'DAYS_EMPLOYED': [days_employed],
            'APPS_CREDIT_INCOME_RATIO': [credit_income_ratio],
            'APPS_EXT_SOURCE_MEAN': [(ext_source_1 + ext_source_2 + ext_source_3) / 3],
            'CNT_FAM_MEMBERS': [fam_members],
            'OWN_CAR_AGE': [np.nan],
            # Behavioral features
            'LIMIT_BAL': [limit_bal],
            'SEX': [sex_val],
            'EDUCATION': [edu_val],
            'MARRIAGE': [mar_val],
            'AGE': [age],
            'PAY_0': [pay_0],
            'PAY_2': [pay_2],
            'PAY_3': [pay_3],
            'PAY_4': [pay_4],
            'PAY_5': [pay_5],
            'PAY_6': [pay_6],
            'BILL_AMT1': [bill_amt1],
            'BILL_AMT2': [bill_amt2],
            'BILL_AMT3': [bill_amt3],
            'BILL_AMT4': [bill_amt4],
            'BILL_AMT5': [bill_amt5],
            'BILL_AMT6': [bill_amt6],
            'PAY_AMT1': [pay_amt1],
            'PAY_AMT2': [pay_amt2],
            'PAY_AMT3': [pay_amt3],
            'PAY_AMT4': [pay_amt4],
            'PAY_AMT5': [pay_amt5],
            'PAY_AMT6': [pay_amt6],
        })

        # Apply feature engineering to both parts
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from feature_engineering import process_apps, behaviorial_features
        
        # Create separate dataframes for traditional and behavioral
        trad_data = input_data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AMT_CREDIT', 
                                 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
                                 'DAYS_BIRTH', 'DAYS_EMPLOYED']].copy()
        
        behav_data = input_data[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                                  'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                                  'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].copy()
        
        # Apply feature engineering
        print(f"\n[HYBRID] Traditional features before engineering: {len(trad_data.columns)}")
        trad_data = process_apps(trad_data)
        print(f"[HYBRID] Traditional features after engineering: {len(trad_data.columns)}")
        
        print(f"[HYBRID] Behavioral features before engineering: {len(behav_data.columns)}")
        behav_data = behaviorial_features(behav_data)
        print(f"[HYBRID] Behavioral features after engineering: {len(behav_data.columns)}")
        
        # Combine engineered features
        input_data = pd.concat([trad_data, behav_data], axis=1)
        print(f"[HYBRID] Total combined features: {len(input_data.columns)}")
        
        return input_data, True
    
    return None, False

def display_prediction_results(predictions, probabilities, model_type):
    """Display prediction results with visualizations"""
    st.markdown("---")
    st.subheader("Prediction Results")
    
    pred = predictions[0]
    prob = probabilities[0] if probabilities is not None else None
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Prediction
        if pred == 1:
            st.error("###  **LIKELY TO DEFAULT**")
        else:
            st.success("###  **UNLIKELY TO DEFAULT**")
        
        if prob is not None:
            risk = classify_risk(prob)
            st.markdown(f"**Risk Classification:** {risk}")
            st.markdown(f"**Default Probability:** {prob * 100:.2f}%")
            
            # Progress bar
            st.progress(prob)
        
        # Show model type
        if model_type == 'ensemble':
            st.info("**Ensemble Model:** Combined Traditional + Behavioral analysis")
        elif model_type == 'traditional':
            st.info(" **Traditional Model:** Home Credit features analysis")
        elif model_type == 'behavioral':
            st.info(" **Behavioral Model:** Payment pattern analysis")
    
    with col2:
        if prob is not None:
            # Gauge chart
            fig = plot_gauge(prob, "Default Probability")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Interpretation
    st.markdown("####  Risk Assessment")
    
    if prob is not None:
        if prob < 0.3:
            st.success("""
            ** Low Risk Applicant**
            - Default probability is low (< 30%)
            - Strong credit indicators
            - **Recommendation:** Approve with standard terms
            """)
        elif prob < 0.6:
            st.warning("""
            ** Medium Risk Applicant**
            - Default probability is moderate (30-60%)
            - Mixed credit indicators
            - **Recommendation:** Approve with careful monitoring or adjusted terms
            """)
        else:
            st.error("""
            ** High Risk Applicant**
            - Default probability is high (> 60%)
            - Weak credit indicators
            - **Recommendation:** Deny or require additional collateral
            """)

if __name__ == "__main__":
    show()
else:
    show()
