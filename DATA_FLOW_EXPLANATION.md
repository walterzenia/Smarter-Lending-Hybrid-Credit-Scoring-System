# 🔄 Data Flow After Input - Single Applicant Prediction

## Current Issue: Missing Feature Engineering! ⚠️

### What Happens Now (INCOMPLETE):

```
User fills form → Submit button clicked → Input data created
                                              ↓
                                   [DataFrame with base features]
                                              ↓
                                       get_predictions()
                                              ↓
                                      align_features()
                          (Tries to match model's expected features)
                                              ↓
                                     MISSING FEATURES!
                          Model expects 31 features, only gets 23
```

---

## The Problem:

### **Behavioral Model Expects 31 Features:**

#### Base Features (11 features) -  We provide these:

1. LIMIT_BAL
2. SEX
3. EDUCATION
4. MARRIAGE
5. AGE
6. PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6

#### Missing: BILL_AMT1-6 and PAY_AMT1-6 (not in form)

#### Engineered Features (20 features) - ❌ NOT CALCULATED:

12. total_billed_amount (sum of BILL_AMT1-6)
13. total_payment_amount (sum of PAY_AMT1-6)
14. avg_transaction_amount (mean of BILL_AMT1-6)
15. max_billed_amount (max of BILL_AMT1-6)
16. max_payment_amount (max of PAY_AMT1-6)
17. spending_volatility (std of BILL_AMT1-6)
18. income_consistency (std of PAY_AMT1-6)
19. bill_change_1_2 (BILL_AMT2 - BILL_AMT1)
20. bill_change_2_3 (BILL_AMT3 - BILL_AMT2)
21. bill_change_3_4 (BILL_AMT4 - BILL_AMT3)
22. bill_change_4_5 (BILL_AMT5 - BILL_AMT4)
23. rolling_balance_volatility (std of bill changes)
24. net_flow_balance (total billed - total payment)
25. debt_stress_index (total billed / total payment)
26. repayment_ratio (total payment / total billed)
27. payment_consistency_ratio (income consistency / total payment)
28. spend_to_income_volatility_ratio (spending volatility / income consistency)
29. max_to_mean_bill_ratio (max billed / avg transaction)
30. missed_payment_count (count of PAY_AMT == 0)
31. credit_utilization_trend (slope of BILL_AMT over time)

---

## What SHOULD Happen:

```
User fills form → Submit button clicked → Input data created
                                              ↓
                                   [DataFrame with base features]
                                              ↓
                                   🔧 APPLY FEATURE ENGINEERING
                              behaviorial_features(input_data)
                                              ↓
                           [DataFrame with 31 complete features]
                                              ↓
                                       get_predictions()
                                              ↓
                                      Model.predict_proba()
                                              ↓
                                   ✅ Predictions + Probabilities
                                              ↓
                                  display_prediction_results()
```

---

## Solutions Needed:

### Option 1: Apply Feature Engineering in Form Function ✅ BEST

**Modify behavioral_input_form():**

```python
def behavioral_input_form():
    # ... collect base features ...

    if submitted:
        # Create base dataframe
        input_data = pd.DataFrame({...})

        # 🔧 APPLY FEATURE ENGINEERING HERE
        from feature_engineering import behaviorial_features
        input_data = behaviorial_features(input_data)

        return input_data, True
```

### Option 2: Apply in get_predictions()  POSSIBLE

**Modify utils.py:**

```python
def get_predictions(model, X):
    # Detect if behavioral model
    if is_behavioral_model(model):
        from feature_engineering import behaviorial_features
        X = behaviorial_features(X)

    # Then proceed with predictions...
```

### Option 3: Create Wrapper Function  ALTERNATIVE

```python
def prepare_features_for_prediction(X, model_type):
    if model_type == 'behavioral':
        from feature_engineering import behaviorial_features
        X = behaviorial_features(X)
    elif model_type == 'traditional':
        from feature_engineering import process_apps
        X = process_apps(X)
    return X
```

---

## Current align_features() Behavior:

The `align_features()` function:

1. Gets expected features from model (31 features)
2. Finds missing features (20 engineered features)
3. **Fills missing with NaN** ❌
4. Replaces NaN with 0 or median ❌

**Result:** Model gets zeros for 20 features instead of calculated values!

---

## Recommended Fix:

**Update behavioral_input_form() to apply feature engineering:**

```python
def behavioral_input_form():
    with st.form("behavioral_form"):
        # ... form fields ...

        submitted = st.form_submit_button(...)

    if submitted:
        # Create base dataframe with BILL_AMT and PAY_AMT columns
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

        # 🔧 APPLY FEATURE ENGINEERING
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from feature_engineering import behaviorial_features

        input_data = behaviorial_features(input_data)

        return input_data, True

    return None, False
```

---

## Same Issue for Traditional Model:

Traditional model expects **487 features** but form only provides **11 base features**.

Missing: 476 engineered features from `process_apps()`!

**The align_features() fills these with zeros/median** → Poor predictions!

---

## Summary:

### Current State:

❌ Base features only (11 traditional, 23 behavioral)  
❌ No feature engineering applied  
❌ Missing features filled with zeros  
❌ Poor prediction accuracy

### After Fix:

✅ Base features collected from form  
✅ Feature engineering applied automatically  
✅ All 31 behavioral / 487 traditional features present  
✅ Accurate predictions

---

## Action Required:

**Update all 3 input form functions to apply feature engineering before returning data!**

1. `traditional_input_form()` → Apply `process_apps()`
2. `behavioral_input_form()` → Apply `behaviorial_features()`
3. `hybrid_input_form()` → Apply both functions
