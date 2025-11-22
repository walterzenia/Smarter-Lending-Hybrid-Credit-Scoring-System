# Loan Default Hybrid System

A comprehensive machine learning system for predicting loan defaults using a hybrid approach that combines traditional credit features with behavioral patterns.

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Ensemble Hybrid Model](#ensemble-hybrid-model)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results & Performance](#results--performance)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Testing & Quality](#testing--quality)
- [Known Limitations](#known-limitations)

---

## Project Overview

This project implements a sophisticated loan default prediction system that leverages **two distinct data sources** to create a powerful hybrid model:

1. **Home Credit Dataset**: Traditional credit features (demographics, credit history, external sources)
2. **UCI Credit Card Dataset**: Behavioral features (payment patterns, spending behavior, credit utilization)

### Key Features

- Multi-model architecture (Traditional, Behavioral, Hybrid)
- Advanced feature engineering pipeline
- Ensemble stacking with meta-learning
- Interactive Streamlit web dashboard
- SHAP-based model interpretability
- Real-time prediction capabilities

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                              │
├──────────────────────┬──────────────────────────────────────┤
│  Home Credit Data    │    UCI Credit Card Data              │
│  • application_train │    • UCI_Credit_Card.csv             │
│  • bureau            │    • uci_interface_test.csv          │
│  • previous_app      │                                      │
│  • installments      │                                      │
│  • pos_cash          │                                      │
│  • credit_card       │                                      │
└──────────────────────┴──────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  src/feature_engineering.py - Core Feature Functions        │
│  • process_apps()        - Application features (13)        │
│  • process_prev()        - Previous loan features           │
│  • process_bureau()      - Credit bureau aggregations       │
│  • process_pos()         - POS cash balance features        │
│  • process_install()     - Installment payment features     │
│  • process_card()        - Credit card features             │
│  • behaviorial_features()- UCI behavioral features (39)     │
│                                                              │
│  src/extract_features.py - Feature Orchestration           │
│  • traditional_features()- Combines apps + prev + bureau    │
│  • hybrid_features()     - All features (487 total)         │
│  • behaviorial_features()- Behavioral pipeline              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Model Training                             │
├──────────────────┬──────────────────┬──────────────────────┤
│  Traditional     │   Behavioral     │   Hybrid Ensemble    │
│  Model           │   Model          │   Model              │
│                  │                  │                      │
│  model_hybrid    │   first_lgbm     │  model_ensemble      │
│  .pkl            │   _model.pkl     │  _wrapper.pkl        │
│                  │                  │                      │
│  487 features    │   31 features    │  Meta-learner        │
│  AUC: ~0.75      │   AUC: ~0.76     │  AUC: 0.8591        │
└──────────────────┴──────────────────┴──────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Streamlit Dashboard (app.py)                    │
├─────────────────────────────────────────────────────────────┤
│  • Home Page              - Project overview                │
│  • EDA Page               - Data exploration                │
│  • Prediction Page        - Batch/Single predictions        │
│  • Feature Importance     - SHAP analysis                   │
│  • Model Metrics          - Performance evaluation          │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### 1. Data Loading

**Location**: `src/data_preprocessing.py`

**Process**:

```python
# Load all Home Credit datasets
apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = get_dataset()

# Load balance-specific datasets
pos, installments, credit_card = get_balance_data()
```

**Key Functions**:

- `get_dataset()`: Loads all 8 CSV files (train, test, previous, bureau, bureau_balance, POS, installments, credit card)
- `get_balance_data()`: Loads the 3 balance history files
- Uses Git LFS for large files (2.7 GB total)
- Concatenates train/test for unified processing

**Results**: Raw datasets loaded and ready for feature engineering

---

### 2. Feature Engineering

**Architecture**: Two-layer feature pipeline

**Layer 1 - Core Functions** (`src/feature_engineering.py`):

- Individual feature transformation functions
- Process specific datasets (apps, bureau, previous, balances)
- Create domain-specific features

**Layer 2 - Orchestration** (`src/extract_features.py`):

- Combines multiple feature sets
- Merges aggregated datasets
- Exports different feature configurations for different models

#### A. Traditional Features (Home Credit)

**Code Reference**:

```python
def process_apps(apps: pd.DataFrame) -> pd.DataFrame:
    """Process application data with engineered features"""

    # External source aggregations
    apps['APPS_EXT_SOURCE_MEAN'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)

    # Credit ratios
    apps['APPS_ANNUITY_CREDIT_RATIO'] = apps['AMT_ANNUITY'] / apps['AMT_CREDIT']
    apps['APPS_CREDIT_INCOME_RATIO'] = apps['AMT_CREDIT'] / apps['AMT_INCOME_TOTAL']

    # Employment ratios
    apps['APPS_EMPLOYED_BIRTH_RATIO'] = apps['DAYS_EMPLOYED'] / apps['DAYS_BIRTH']
    apps['APPS_INCOME_EMPLOYED_RATIO'] = apps['AMT_INCOME_TOTAL'] / apps['DAYS_EMPLOYED']

    return apps
```

**Feature Categories**:

1. **External Source Features** (3 features)
   - Mean and standard deviation of external credit scores
2. **Financial Ratios** (8 features)
   - Annuity/Credit, Credit/Income, Goods/Credit ratios
   - Income distribution across family members
3. **Temporal Ratios** (5 features)

   - Employment/Birth, Income/Employed, Car age ratios

4. **Bureau Features** (40+ features)

   - Credit history aggregations
   - DPD (Days Past Due) indicators
   - Active credit statistics

5. **Previous Application Features** (30+ features)
   - Historical application patterns
   - Approval/Refusal ratios
   - Credit differences and interest rates

**Total Traditional Features**: 487

---

#### B. Behavioral Features (UCI Credit Card)

**Code Reference**:

```python
def behaviorial_features(uci: pd.DataFrame) -> pd.DataFrame:
    """Engineer behavioral features from payment patterns"""

    # AGGREGATE FEATURES
    uci["total_billed_amount"] = uci[["BILL_AMT1", ..., "BILL_AMT6"]].sum(axis=1)
    uci["total_payment_amount"] = uci[["PAY_AMT1", ..., "PAY_AMT6"]].sum(axis=1)
    uci["avg_transaction_amount"] = uci[["BILL_AMT1", ..., "BILL_AMT6"]].mean(axis=1)

    # VOLATILITY INDICATORS
    uci["spending_volatility"] = uci[["BILL_AMT1", ..., "BILL_AMT6"]].std(axis=1)
    uci["income_consistency"] = uci[["PAY_AMT1", ..., "PAY_AMT6"]].std(axis=1)

    # FINANCIAL STRESS INDICATORS
    uci["net_flow_balance"] = uci["total_billed_amount"] - uci["total_payment_amount"]
    uci["debt_stress_index"] = uci["total_billed_amount"] / (uci["total_payment_amount"] + 1)
    uci["repayment_ratio"] = uci["total_payment_amount"] / (uci["total_billed_amount"] + 1)

    # BEHAVIORAL PATTERNS
    uci["missed_payment_count"] = (uci[["PAY_AMT1", ..., "PAY_AMT6"]] == 0).sum(axis=1)

    # TREND ANALYSIS
    uci["credit_utilization_trend"] = compute_slope(uci[bill_columns])

    return uci
```

**Feature Categories**:

1. **Aggregate Metrics** (5 features)

   - Total billed, total payment, averages, maximums

2. **Volatility Measures** (3 features)

   - Spending volatility, income consistency, rolling balance changes

3. **Financial Stress** (3 features)

   - Net flow, debt stress index, repayment ratio

4. **Behavioral Ratios** (3 features)

   - Payment consistency, spend-to-income volatility, max-to-mean ratios

5. **Payment Behavior** (2 features)
   - Missed payment count, credit utilization trend

**Total Behavioral Features**: 31 (including base UCI features)

---

## Model Development

### Model 1: Traditional Model

**File**: `models/model_hybrid.pkl`

**Features**: 487 traditional features from Home Credit data

**Architecture**: LightGBM Classifier

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}
```

**Performance**:

- Training Set: ~0.85 AUC
- Validation Set: ~0.75 AUC
- Test Set: ~0.74 AUC

**Key Strengths**:

- Captures credit history patterns
- Strong on traditional lending factors
- Good generalization

---

### Model 2: Behavioral Model

**File**: `models/first_lgbm_model.pkl`

**Features**: 31 behavioral features from UCI Credit Card dataset

**Architecture**: LightGBM Classifier

**Performance**:

- Training Set: ~0.82 AUC
- Validation Set: ~0.76 AUC
- Test Set: ~0.75 AUC

**Key Strengths**:

- Identifies spending patterns
- Captures payment behavior
- Detects financial stress signals

---

### Model 3: Ensemble Hybrid Model

**File**: `models/model_ensemble_wrapper.pkl`

**Training Script**: `src/train_ensemble_hybrid.py`

> ** For detailed ensemble framework explanation, see [HYBRID_MODEL_SUMMARY.md](HYBRID_MODEL_SUMMARY.md)**
>
> The document includes:
>
> - Comprehensive stacking architecture with meta-learning
> - Comparison with other ensemble methods (Bagging, Boosting, Voting)
> - Two-layer design with 27 meta-features
> - Training process and theoretical justification
> - Performance analysis showing +14% AUC improvement

#### Architecture: Stacking Ensemble

```
Level 0 (Base Models):
├─ Traditional Model (model_hybrid.pkl)
│  └─ 487 features → probability_traditional
└─ Behavioral Model (first_lgbm_model.pkl)
   └─ 31 features → probability_behavioral

Level 1 (Meta Features):
├─ pred_traditional
├─ pred_behavioral
├─ pred_avg = (pred_trad + pred_behav) / 2
├─ pred_max = max(pred_trad, pred_behav)
├─ pred_min = min(pred_trad, pred_behav)
├─ pred_diff = |pred_trad - pred_behav|
└─ pred_ratio = pred_trad / pred_behav
   + Top 10 features from each base model

Level 2 (Meta Learner):
└─ LightGBM Meta-Model
   └─ Final prediction
```

**Training Process**:

```python
# 1. Generate meta-features from base models
pred_traditional = model_traditional.predict_proba(X_traditional)[:, 1]
pred_behavioral = model_behavioral.predict_proba(X_behavioral)[:, 1]

# 2. Create meta-feature matrix
meta_features = create_meta_features(pred_traditional, pred_behavioral)

# 3. Train meta-learner
meta_model = lgb.train(params, meta_features, y_target)

# 4. Combine into ensemble wrapper
ensemble = EnsembleHybridModel(meta_model, model_trad, model_behav)
```

**Performance**:

```
AUC-ROC: 0.8591

Classification Report:
              precision    recall  f1-score   support
         0.0       0.93      1.00      0.96      3198
         1.0       0.66      0.09      0.16       270

    accuracy                           0.93      3468

Confusion Matrix:
[[3185   13]
 [ 245   25]]
```

**Key Improvements**:

- +9% AUC improvement over traditional model
- +9.1% AUC improvement over behavioral model
- Better false positive reduction
- Robust to feature distribution shifts

---

## Hybrid Feature Creation

**Script**: `src/create_hybrid_features.py`

This script bridges the gap between the two datasets by simulating missing features:

### For Home Credit Users (smoke.csv):

```python
def simulate_behavioral_features_for_smoke(smoke_df):
    """Simulate UCI-style behavioral features"""

    # Simulate base UCI features
    behavioral_sim['LIMIT_BAL'] = smoke_df['AMT_CREDIT'] * random(0.5, 1.5)
    behavioral_sim['SEX'] = smoke_df['CODE_GENDER'].map({'M': 1, 'F': 2})
    behavioral_sim['AGE'] = (-smoke_df['DAYS_BIRTH'] / 365).astype(int)

    # Simulate payment status
    behavioral_sim['PAY_0'] to behavioral_sim['PAY_6'] = simulate_payment_history()

    # Simulate bills and payments
    behavioral_sim['BILL_AMT1'] to ['BILL_AMT6'] = simulate_from_credit_income()
    behavioral_sim['PAY_AMT1'] to ['PAY_AMT6'] = simulate_from_annuity()

    # Apply behavioral feature engineering
    return behaviorial_features(behavioral_sim)
```

### For UCI Users (uci_interface_test.csv):

```python
def simulate_traditional_features_for_uci(uci_df):
    """Simulate Home Credit-style traditional features"""

    # Demographics
    traditional_sim['DAYS_BIRTH'] = -(uci_df['AGE'] * 365)
    traditional_sim['AMT_INCOME_TOTAL'] = uci_df['LIMIT_BAL'] * random(2, 6)

    # Credit amounts
    traditional_sim['AMT_CREDIT'] = uci_df['LIMIT_BAL'] * random(0.3, 0.9)
    traditional_sim['AMT_ANNUITY'] = AMT_CREDIT / random(12, 60)

    # External sources (credit scores)
    traditional_sim['EXT_SOURCE_1'] to ['EXT_SOURCE_3'] = random(0.2, 0.8)

    # Apply traditional feature engineering
    return process_apps(traditional_sim)
```

**Output Datasets**:

- `data/smoke_hybrid_features.csv`: 20,000 rows × 527 columns
- `data/uci_hybrid_features.csv`: 1,425 rows × 57 columns

---

## Streamlit Dashboard

**Main File**: `app.py`

**Structure**: Multi-page application using Streamlit's native page routing

### Available Pages:

1. **Home Page** (`pages/0_Home.py`) - Project overview and navigation
2. **EDA Page** (`pages/1_EDA.py`) - Exploratory data analysis with interactive charts
3. **Prediction Page** (`pages/2_Prediction.py`) - Batch and single predictions
4. **Feature Importance** (`pages/3_Feature_Importance.py`) - SHAP analysis
5. **Model Metrics** (`pages/4_Model_Metrics.py`) - Performance evaluation

### Key Features:

- **Interactive Visualizations**: Plotly charts for all visualizations
- **Multiple Model Support**: Traditional, Behavioral, and Ensemble models
- **Batch Processing**: Upload CSV for bulk predictions
- **Single Predictions**: Manual input with sliders
- **Risk Classification**: Low , Medium , High
- **Downloadable Results**: Export predictions and metrics

---

## Installation & Setup

### Prerequisites

```bash
Python 3.8+
pip
virtualenv (recommended)
```

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd "Loan Default Hybrid System"
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv myenv

# Activate (Windows PowerShell)
myenv\Scripts\Activate.ps1

# Activate (Windows Command Prompt)
myenv\Scripts\activate.bat

# Activate (Linux/Mac)
source myenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirement.txt
```

**Key Dependencies**:

- streamlit>=1.28.0
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0
- lightgbm>=4.0.0
- plotly>=5.17.0
- shap>=0.42.0
- joblib>=1.3.0

### Step 4: Verify Data Files

Ensure these files exist in `data/`:

- `smoke_engineered.csv`
- `UCI_Credit_Card.csv`
- `uci_interface_test.csv`
- `smoke_hybrid_features.csv`

### Step 5: Verify Model Files

Ensure these files exist in `models/`:

- `model_hybrid.pkl`
- `first_lgbm_model.pkl`
- `model_ensemble_wrapper.pkl`

---

## Usage

### Running the Streamlit Dashboard

```powershell
# Activate virtual environment
myenv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run app.py
```

Access the dashboard at: **http://localhost:8501**

---

### Creating Hybrid Features

```powershell
# Generate hybrid feature datasets
python src/create_hybrid_features.py
```

**Output**:

- `data/smoke_hybrid_features.csv`
- `data/uci_hybrid_features.csv`

---

### Training Ensemble Model

```powershell
# Train the ensemble hybrid model
python src/train_ensemble_hybrid.py
```

**Output**:

- `models/model_ensemble_hybrid.pkl` - Meta-learner
- `models/model_ensemble_wrapper.pkl` - Complete ensemble
- `models/ensemble_metadata.pkl` - Feature metadata

---

### Making Predictions Programmatically

#### Using Ensemble Model:

```python
import joblib
import pandas as pd

# Load ensemble
ensemble = joblib.load('models/model_ensemble_wrapper.pkl')

# Load hybrid data
df = pd.read_csv('data/smoke_hybrid_features.csv')

# Predict
probabilities = ensemble.predict_proba(df)[:, 1]
predictions = ensemble.predict(df)

# Risk classification
def classify_risk(prob):
    if prob < 0.3: return "Low Risk "
    elif prob < 0.6: return "Medium Risk "
    else: return "High Risk "

risks = [classify_risk(p) for p in probabilities]
```

---

## Results & Performance

### Model Comparison

| Model               | Features | AUC-ROC    | Precision | Recall   | F1-Score | Use Case                      |
| ------------------- | -------- | ---------- | --------- | -------- | -------- | ----------------------------- |
| **Traditional**     | 487      | 0.7500     | 0.68      | 0.45     | 0.54     | Standard credit assessment    |
| **Behavioral**      | 31       | 0.7600     | 0.71      | 0.42     | 0.53     | Payment behavior analysis     |
| **Ensemble Hybrid** | 518      | **0.8591** | **0.66**  | **0.09** | **0.16** | Comprehensive risk assessment |

### Performance Highlights

#### Ensemble Model (Best Performance):

```
AUC-ROC: 0.8591

Confusion Matrix:
                Predicted Negative    Predicted Positive
Actual Negative       3185                   13
Actual Positive        245                   25

Metrics:
- True Negative Rate (Specificity): 99.6%
- True Positive Rate (Sensitivity): 9.3%
- False Positive Rate: 0.4%
- Precision (Positive): 66%
```

**Interpretation**:

- **Excellent at identifying non-defaulters** (99.6% specificity)
- **Very few false alarms** (only 13 false positives out of 3198)
- **Conservative on defaults** (catches 9.3% of true defaults)
- **Suitable for**: Low-risk lending where minimizing false positives is critical

---

### Feature Importance (Top 10)

#### Traditional Model:

1. `EXT_SOURCE_2` - External credit score
2. `EXT_SOURCE_3` - External credit score
3. `DAYS_BIRTH` - Age of applicant
4. `AMT_CREDIT` - Loan amount
5. `APPS_EXT_SOURCE_MEAN` - Avg external score
6. `AMT_ANNUITY` - Monthly payment
7. `AMT_GOODS_PRICE` - Price of goods
8. `DAYS_EMPLOYED` - Employment duration
9. `AMT_INCOME_TOTAL` - Total income
10. `APPS_CREDIT_INCOME_RATIO` - Credit/income ratio

#### Behavioral Model:

1. `PAY_0` - Most recent payment status
2. `PAY_2` - Payment status 2 months ago
3. `LIMIT_BAL` - Credit limit
4. `total_payment_amount` - Total payments made
5. `debt_stress_index` - Bills/payments ratio
6. `PAY_3` - Payment status 3 months ago
7. `repayment_ratio` - Payment/bill ratio
8. `missed_payment_count` - Number of missed payments
9. `spending_volatility` - Variation in spending
10. `credit_utilization_trend` - Utilization slope

---

## Project Structure

```
Loan Default Hybrid System/
│
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── HYBRID_MODEL_SUMMARY.md        # Detailed ensemble architecture
├── DEPLOYMENT_GUIDE.md            # Deployment instructions
│
├── data/                          # Data directory (tracked with Git LFS)
│   ├── application_train.csv      # Home Credit training data (2.5 GB)
│   ├── smoke_engineered.csv       # Processed holdout data (20K rows)
│   ├── smoke_hybrid_features.csv  # Hybrid features for ensemble
│   ├── UCI_Credit_Card.csv        # UCI behavioral data
│   ├── uci_interface_test.csv     # UCI test interface
│   └── bureau.csv, previous_application.csv, etc.
│
├── models/                        # Trained models
│   ├── model_hybrid.pkl           # Traditional model (7.69 MB, 487 features)
│   ├── first_lgbm_model.pkl       # Behavioral model (1.05 MB, 31 features)
│   ├── model_ensemble_wrapper.pkl # Ensemble wrapper (8.91 MB)
│   ├── model_ensemble_hybrid.pkl  # Raw meta-learner
│   └── ensemble_metadata.pkl      # Ensemble configuration
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── config.py                  # Configuration and file paths
│   │
│   ├── data_preprocessing.py      # Data loading and validation
│   │   └── get_dataset()          # Loads all Home Credit CSVs
│   │   └── get_balance_data()     # Loads balance histories
│   │
│   ├── feature_engineering.py     # Core feature transformation functions
│   │   └── process_apps()         # Application features (13)
│   │   └── process_prev()         # Previous loan features
│   │   └── process_bureau()       # Bureau aggregations
│   │   └── process_pos()          # POS cash features
│   │   └── process_install()      # Installment features
│   │   └── process_card()         # Credit card features
│   │   └── behaviorial_features() # UCI behavioral pipeline
│   │
│   ├── extract_features.py        # Feature orchestration layer
│   │   └── traditional_features() # Combines apps + prev + bureau (487)
│   │   └── hybrid_features()      # All feature sets combined (487)
│   │   └── behaviorial_features() # Behavioral pipeline wrapper
│   │
│   ├── model_training.py          # Model training pipeline
│   │   └── train_classifier()     # LightGBM training with CV
│   │
│   ├── train_ensemble_hybrid.py   # Ensemble training script
│   │   └── Creates meta-learner with stacking
│   │
│   ├── create_hybrid_features.py  # Feature simulation for ensemble
│   │   └── Generates behavioral features for Home Credit data
│   │
│   ├── ensemble_model.py          # Ensemble wrapper class
│   │   └── EnsembleHybridModel    # Production-ready ensemble
│   │
│   ├── inference.py               # Prediction utilities
│   ├── model_evaluation.py        # Model evaluation metrics
│   ├── utils.py                   # Helper functions
│   └── visualization.py           # Plotting utilities
│   ├── smoke_hybrid_features.csv  # Hybrid features (Home Credit)
│   └── uci_hybrid_features.csv    # Hybrid features (UCI)
│
├── models/                        # Trained models
│   ├── model_hybrid.pkl           # Traditional model (487 features)
│   ├── first_lgbm_model.pkl       # Behavioral model (31 features)
│   ├── model_ensemble_hybrid.pkl  # Meta-learner (stacking)
│   ├── model_ensemble_wrapper.pkl # Complete ensemble
│   └── ensemble_metadata.pkl      # Feature metadata
│
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data cleaning
│   ├── feature_engineering.py     # Feature engineering
│   ├── create_hybrid_features.py  # Hybrid feature generation
│   └── train_ensemble_hybrid.py   # Ensemble training
│
├── apps/                          # Streamlit utilities
│   └── utils.py                   # Helper functions
│
├── pages/                         # Streamlit pages
│   ├── 0_Home.py                 # Landing page
│   ├── 1_EDA.py                  # Data exploration
│   ├── 2_Prediction.py           # Predictions
│   ├── 3_Feature_Importance.py   # SHAP analysis
│   └── 4_Model_Metrics.py        # Performance metrics
│
└── myenv/                        # Virtual environment
```

---

## Documentation

### User Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user guide for the Streamlit dashboard
  - Getting started and setup
  - Making predictions (manual and batch)
  - Understanding results and risk levels
  - Model metrics interpretation
  - Troubleshooting common issues

### Technical Documentation

- **[HYBRID_MODEL_SUMMARY.md](HYBRID_MODEL_SUMMARY.md)** - Detailed ensemble architecture and stacking framework
- **[MODEL_ARCHITECTURE_FLOWCHART.md](MODEL_ARCHITECTURE_FLOWCHART.md)** - Visual architecture guide
- **[DATA_FLOW_EXPLANATION.md](DATA_FLOW_EXPLANATION.md)** - Data pipeline documentation
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment instructions for multiple platforms

### Project Status & Limitations

- **[PROJECT_LIMITATIONS.md](PROJECT_LIMITATIONS.md)** - Known limitations and constraints
  - Data limitations
  - Model constraints
  - Feature simulation assumptions
  - Performance boundaries

---

## Testing & Quality

### Test Suite

**Location**: `tests/` directory

**Test Files**:

- `test_ensemble_direct.py` - Direct ensemble model testing
- `test_ensemble_streamlit.py` - Streamlit integration tests
- `test_high_risk_predictions.py` - High-risk scenario validation
- `test_traditional_prediction.py` - Traditional model tests

### Running Tests

```powershell
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ensemble_direct.py

# Run with coverage
pytest --cov=src tests/
```

---

## Known Limitations

### Model Limitations

**Class Imbalance Impact:**

- Low recall (9.3%) for default class
- Model optimized for minimizing false positives
- May miss true defaults in edge cases

**Feature Simulation:**

- Hybrid features are simulated when not natively available
- Simulation based on statistical relationships
- Real hybrid data would improve accuracy

**Training Data:**

- Ensemble trained on 20,000 samples (smoke_hybrid_features.csv)
- May not generalize to all populations
- Performance varies by demographic segments

### Data Constraints

**Traditional Model (Home Credit):**

- Requires 487 features - high data collection burden
- Some features may not be available for all applicants
- External credit scores (EXT_SOURCE) are critical but proprietary

**Behavioral Model (UCI):**

- Requires 6 months of payment history
- New customers cannot be scored
- Transaction data must be formatted consistently

**Ensemble Model:**

- Needs both traditional AND behavioral data
- Higher computational cost
- More complex deployment requirements

### Performance Considerations

**Prediction Speed:**

- Single prediction: ~100-200ms
- Batch (1000 rows): ~5-10 seconds
- Ensemble slower than individual models due to meta-learning

**Memory Usage:**

- Models total: ~200MB disk space
- Runtime memory: ~500MB-1GB
- Large batch predictions may require more RAM

### Deployment Notes

**Feature Alignment:**

- Input data must exactly match training feature names
- Missing columns will cause prediction failure
- Extra columns are ignored

**Version Compatibility:**

- Models trained with scikit-learn 1.6.1
- Running with 1.7.2 works but version warnings occur
- Retraining recommended for long-term production use

For detailed limitations, see **[PROJECT_LIMITATIONS.md](PROJECT_LIMITATIONS.md)**

---

## Technical Deep Dive

### Why Stacking Works Here

**Problem**: Traditional and behavioral models capture different aspects of credit risk

- Traditional → Static borrower characteristics
- Behavioral → Dynamic payment patterns

**Solution**: Stacking learns optimal weights and interactions between models

**Result**: 9% AUC improvement by capturing complementary information

---

### Handling Class Imbalance

**Challenge**: Only 8% of loans default in the dataset

**Solutions Implemented**:

1. **Stratified Sampling**: Preserve class distribution in train/test splits
2. **Class Weights**: Penalize false negatives more heavily
3. **Threshold Tuning**: Adjust classification threshold based on business costs
4. **Evaluation Metrics**: Focus on AUC-ROC (robust to imbalance)

---

## Key Learnings

1. **Feature Engineering is Crucial**: Engineered features (ratios, aggregations) outperformed raw features
2. **Ensemble Methods Add Value**: Stacking captured complementary information from different data sources
3. **Handle Imbalance Carefully**: Class weights and stratified sampling are essential
4. **Interpretability Matters**: SHAP values crucial for stakeholder trust
5. **Production Readiness**: Feature alignment critical for deployment

---

## Future Enhancements

### Short Term

- [ ] Improve recall for default class
- [ ] Add API endpoint for programmatic access
- [ ] Implement hyperparameter tuning
- [ ] Add more visualizations

### Long Term

- [ ] Deep learning for unstructured data
- [ ] Automated retraining pipeline
- [ ] Real-time feature computation
- [ ] Causal inference for policy interventions

---

## Acknowledgments

### Data Sources

- **Home Credit Group**: Home Credit Default Risk dataset (Kaggle)
- **UCI Machine Learning Repository**: Default of Credit Card Clients dataset

### Libraries & Tools

- **LightGBM**: Microsoft's gradient boosting framework
- **Streamlit**: Interactive dashboard framework
- **SHAP**: Model interpretability
- **Plotly**: Interactive visualizations
- **scikit-learn**: Machine learning utilities

---

**Last Updated**: November 18, 2025  
**Version**: 2.0.0  
**Status**: Production Ready

---
