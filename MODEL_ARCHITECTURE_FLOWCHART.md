### Loan Default Hybrid System - Model Architecture Flowchart

#### System Overview



```mermaid
graph TB
    Start([User Input]) --> Choice{Prediction Mode}
    Choice -->|Batch| CSV[Upload CSV File]
    Choice -->|Manual| Form[Fill Input Form]

    CSV --> ModelSelect{Select Model Type}
    Form --> ModelSelect

    ModelSelect -->|Traditional| TradModel[Traditional Model<br/>model_hybrid.pkl]
    ModelSelect -->|Behavioral| BehavModel[Behavioral Model<br/>first_lgbm_model.pkl]
    ModelSelect -->|Ensemble| EnsModel[Ensemble Model<br/>model_ensemble_wrapper.pkl]

    TradModel --> Predict[Make Prediction]
    BehavModel --> Predict
    EnsModel --> Predict

    Predict --> Results[Display Results]
    Results --> End([Risk Assessment])

    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style TradModel fill:#fff9c4
    style BehavModel fill:#f8bbd0
    style EnsModel fill:#ce93d8
```

---

### Detailed Model Pipeline

**1. Traditional Model Pipeline (Home Credit Features)**

```mermaid
graph LR
    A[Raw Input<br/>11 Base Features] --> B[Feature Engineering<br/>process_apps]
    B --> C[Engineered Data<br/>487 Features]
    C --> D[Traditional Model<br/>LightGBM]
    D --> E[Prediction Output<br/>Default Probability]

    subgraph "Base Features"
    A1[EXT_SOURCE 1-3<br/>Credit Scores]
    A2[AMT_CREDIT<br/>AMT_INCOME<br/>AMT_ANNUITY]
    A3[DAYS_BIRTH<br/>DAYS_EMPLOYED]
    A4[CNT_FAM_MEMBERS<br/>OWN_CAR_AGE]
    end

    subgraph "Engineered Features Examples"
    C1[Credit-to-Income Ratio]
    C2[Mean External Score]
    C3[Employment-to-Age Ratio]
    C4[Payment Capacity]
    C5[+ 483 More Features]
    end

    style A fill:#bbdefb
    style C fill:#c5e1a5
    style D fill:#fff59d
    style E fill:#ffccbc
```

**Traditional Model Flow:**

1. **Input**: 11 base features from Home Credit dataset
2. **Processing**: `process_apps()` function creates 487 engineered features
3. **Model**: LightGBM trained on 487 features
4. **Output**: Default probability (0-1)

---

**2. Behavioral Model Pipeline (UCI Credit Card Features)**

```mermaid
graph LR
    A[Raw Input<br/>23 Base Features] --> B[Feature Engineering<br/>behaviorial_features]
    B --> C[Engineered Data<br/>31 Features]
    C --> D[Behavioral Model<br/>LightGBM]
    D --> E[Prediction Output<br/>Default Probability]

    subgraph "Base Features"
    A1[LIMIT_BAL<br/>Credit Limit]
    A2[Demographic<br/>SEX, EDU, MARRIAGE, AGE]
    A3[Payment History<br/>PAY_0 to PAY_6]
    A4[Bill Amounts<br/>BILL_AMT1-6]
    A5[Payment Amounts<br/>PAY_AMT1-6]
    end

    subgraph "Engineered Features Examples"
    C1[total_billed_amount]
    C2[total_payment_amount]
    C3[spending_volatility]
    C4[payment_consistency_ratio]
    C5[debt_stress_index]
    C6[+ 8 More Features]
    end

    style A fill:#f8bbd0
    style C fill:#c5e1a5
    style D fill:#f48fb1
    style E fill:#ffccbc
```

**Behavioral Model Flow:**

1. **Input**: 23 base features from UCI Credit Card dataset
2. **Processing**: `behaviorial_features()` function creates 31 total features (23 base + 8 engineered)
3. **Model**: LightGBM trained on 31 features
4. **Output**: Default probability (0-1)

---

**3. Ensemble Model Pipeline (Hybrid Features)**

```mermaid
graph TB
    Start[Raw Input<br/>Combined Features] --> Split{Split Features}

    Split --> TradPath[Traditional Features<br/>11 Base]
    Split --> BehavPath[Behavioral Features<br/>23 Base]

    TradPath --> TradEng[process_apps<br/>Feature Engineering]
    BehavPath --> BehavEng[behaviorial_features<br/>Feature Engineering]

    TradEng --> TradOut[487 Traditional<br/>Features]
    BehavEng --> BehavOut[31 Behavioral<br/>Features]

    TradOut --> Combine[Concatenate<br/>Features]
    BehavOut --> Combine

    Combine --> Hybrid[Hybrid Dataset<br/>518 Total Features]
    Hybrid --> Ensemble[Ensemble Model<br/>model_ensemble_wrapper.pkl]
    Ensemble --> Result[Final Prediction<br/>Default Probability]

    style Start fill:#e1bee7
    style TradPath fill:#bbdefb
    style BehavPath fill:#f8bbd0
    style Hybrid fill:#c5e1a5
    style Ensemble fill:#ce93d8
    style Result fill:#ffccbc
```

**Ensemble Model Flow:**

1. **Input**: Combined 34 base features (11 traditional + 23 behavioral)
2. **Split**: Separate features into traditional and behavioral groups
3. **Processing**:
   - Apply `process_apps()` to traditional features → 487 features
   - Apply `behaviorial_features()` to behavioral features → 31 features
4. **Combine**: Concatenate both feature sets → 518 total features
5. **Model**: Ensemble LightGBM trained on hybrid features
6. **Output**: Default probability (0-1)

---

## Feature Engineering Details

### Traditional Feature Engineering (`process_apps`)

```mermaid
graph TD
    Input[11 Base Features] --> FE[Feature Engineering]

    FE --> Cat1[Ratio Features<br/>Credit/Income<br/>Annuity/Credit]
    FE --> Cat2[Aggregation Features<br/>Mean External Source<br/>Total Credit Products]
    FE --> Cat3[Temporal Features<br/>Age from Birth Date<br/>Employment Duration]
    FE --> Cat4[Statistical Features<br/>Min/Max/Mean/Std<br/>Across Categories]
    FE --> Cat5[Interaction Features<br/>Cross-feature Products<br/>Conditional Values]

    Cat1 --> Output[487 Engineered Features]
    Cat2 --> Output
    Cat3 --> Output
    Cat4 --> Output
    Cat5 --> Output

    style Input fill:#bbdefb
    style FE fill:#fff59d
    style Output fill:#c5e1a5
```

#### Behavioral Feature Engineering (`behaviorial_features`)

```mermaid
graph TD
    Input[23 Base Features] --> FE[Feature Engineering]

    FE --> Cat1[Financial Aggregates<br/>Total Billed Amount<br/>Total Payment Amount<br/>Avg Transaction]
    FE --> Cat2[Volatility Metrics<br/>Spending Volatility<br/>Rolling Balance<br/>Income Consistency]
    FE --> Cat3[Payment Behavior<br/>Payment Consistency<br/>Repayment Ratio<br/>Missed Payment Count]
    FE --> Cat4[Risk Indicators<br/>Debt Stress Index<br/>Credit Utilization<br/>Spend-to-Income Ratio]
    FE --> Cat5[Trend Features<br/>Bill Changes (1-2, 3-4, 4-5)<br/>Credit Utilization Trend]

    Cat1 --> Output[31 Total Features<br/>23 Base + 8 Engineered]
    Cat2 --> Output
    Cat3 --> Output
    Cat4 --> Output
    Cat5 --> Output

    style Input fill:#f8bbd0
    style FE fill:#f48fb1
    style Output fill:#c5e1a5
```

---

### Data Flow Architecture

#### Manual Input (Single Applicant)

```mermaid
sequenceDiagram
    participant U as User
    participant F as Input Form
    participant FE as Feature Engineering
    participant M as Model
    participant R as Results Display

    U->>F: Fill Applicant Details
    F->>FE: Submit Base Features

    alt Traditional Model
        FE->>FE: process_apps(11 features)
        FE->>M: Send 487 features
    else Behavioral Model
        FE->>FE: behaviorial_features(23 features)
        FE->>M: Send 31 features
    else Ensemble Model
        FE->>FE: process_apps(11) + behaviorial_features(23)
        FE->>M: Send 518 features
    end

    M->>M: LightGBM Prediction
    M->>R: Return Probability
    R->>U: Display Risk Assessment
```

#### Batch Input (CSV Upload)

```mermaid
sequenceDiagram
    participant U as User
    participant CSV as CSV File
    participant P as Preprocessor
    participant M as Model
    participant R as Results Export

    U->>CSV: Upload Dataset
    CSV->>P: Load Data (N rows)
    P->>P: Validate Features

    alt Features Already Engineered
        P->>M: Pass Data Directly
    else Raw Features
        P->>P: Apply Feature Engineering
        P->>M: Send Engineered Features
    end

    M->>M: Batch Predictions (N rows)
    M->>R: Return N Probabilities
    R->>U: Download Predictions CSV
```

---

### Model Selection Logic

```mermaid
graph TD
    Start([Model Selection]) --> Check{Check Model Name}

    Check -->|Contains 'hybrid'| Trad[Traditional Model Type]
    Check -->|Contains 'lgbm'| Behav[Behavioral Model Type]
    Check -->|Contains 'ensemble'| Ens[Ensemble Model Type]

    Trad --> TradForm[Show Traditional Form<br/>11 Base Features]
    Behav --> BehavForm[Show Behavioral Form<br/>23 Base Features]
    Ens --> EnsForm[Show Hybrid Form<br/>34 Combined Features]

    TradForm --> TradFE[Apply process_apps]
    BehavForm --> BehavFE[Apply behaviorial_features]
    EnsForm --> EnsFE[Apply Both Functions]

    TradFE --> Predict[Make Prediction]
    BehavFE --> Predict
    EnsFE --> Predict

    style Start fill:#e1f5ff
    style Trad fill:#fff9c4
    style Behav fill:#f8bbd0
    style Ens fill:#ce93d8
    style Predict fill:#c8e6c9
```

---

### Risk Classification Pipeline

```mermaid
graph LR
    A[Model Output<br/>Probability 0-1] --> B{Classify Risk}

    B -->|< 0.3| Low[Low Risk<br/>Approve Standard]
    B -->|0.3 - 0.6| Med[Medium Risk<br/>Approve with Monitoring]
    B -->|> 0.6| High[High Risk<br/>Deny/Require Collateral]

    Low --> Display[Display Results]
    Med --> Display
    High --> Display

    Display --> Viz[Risk Visualization]

    Viz --> G1[Gauge Chart]
    Viz --> G2[Progress Bar]
    Viz --> G3[Risk Metrics]

    style A fill:#e1f5ff
    style Low fill:#c8e6c9
    style Med fill:#fff9c4
    style High fill:#ffccbc
    style Display fill:#e1bee7
```

---

### Feature Count Summary

| Model Type      | Base Features | Engineered Features | Total Features |
| --------------- | ------------- | ------------------- | -------------- |
| **Traditional** | 11            | 476                 | **487**        |
| **Behavioral**  | 23            | 8                   | **31**         |
| **Ensemble**    | 34 (11+23)    | 484 (476+8)         | **518**        |

---

### Key Engineering Functions

#### Traditional: `process_apps(df)`

```python
Input:  11 base features
        ├── EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
        ├── AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY, AMT_GOODS_PRICE
        ├── DAYS_BIRTH, DAYS_EMPLOYED
        └── CNT_FAM_MEMBERS, OWN_CAR_AGE

Output: 487 engineered features
        ├── Original 11 features
        ├── 476 calculated features:
            ├── Ratios (credit/income, annuity/income, etc.)
            ├── Aggregations (mean, sum, max, min)
            ├── Statistical (std, variance, percentiles)
            └── Interactions (cross-products, conditionals)
```

#### Behavioral: `behaviorial_features(df)`

```python
Input:  23 base features
        ├── LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
        ├── PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
        ├── BILL_AMT1-6 (6 features)
        └── PAY_AMT1-6 (6 features)

Output: 31 features (23 base + 8 engineered)
        ├── Original 23 features
        ├── 8 calculated features:
            ├── total_billed_amount
            ├── total_payment_amount
            ├── avg_transaction_amount
            ├── spending_volatility
            ├── payment_consistency_ratio
            ├── debt_stress_index
            ├── credit_utilization_trend
            └── missed_payment_count (+ others)
```

---

#### Model Training Architecture

```mermaid
graph TB
    subgraph "Training Phase (Offline)"
        T1[Historical Data] --> T2[Feature Engineering]
        T2 --> T3[Train-Test Split]
        T3 --> T4[LightGBM Training]
        T4 --> T5[Save Model .pkl]
    end

    subgraph "Inference Phase (Runtime)"
        I1[New Applicant Data] --> I2[Feature Engineering]
        I2 --> I3[Load Model .pkl]
        I3 --> I4[Predict]
        I4 --> I5[Return Probability]
    end

    T5 -.Model Files.-> I3

    style T1 fill:#e1f5ff
    style T5 fill:#c8e6c9
    style I1 fill:#fff9c4
    style I5 fill:#ffccbc
```

---

#### System Architecture Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Streamlit UI<br/>app.py]
        Pages[Pages<br/>Prediction.py<br/>EDA.py<br/>Model_Metrics.py]
    end

    subgraph "Business Logic Layer"
        Utils[Utilities<br/>apps/utils.py]
        FE[Feature Engineering<br/>src/feature_engineering.py]
    end

    subgraph "Model Layer"
        M1[model_hybrid.pkl<br/>487 features]
        M2[first_lgbm_model.pkl<br/>31 features]
        M3[model_ensemble_wrapper.pkl<br/>518 features]
    end

    subgraph "Data Layer"
        D1[CSV Files]
        D2[Manual Input]
    end

    UI --> Pages
    Pages --> Utils
    Utils --> FE
    FE --> M1
    FE --> M2
    FE --> M3
    D1 --> Pages
    D2 --> Pages

    style UI fill:#e1f5ff
    style Utils fill:#fff9c4
    style FE fill:#c5e1a5
    style M1 fill:#fff59d
    style M2 fill:#f48fb1
    style M3 fill:#ce93d8
```

---

---

#### Complete System Architecture

##### Full Application Structure

```mermaid
graph TB
    subgraph "Presentation Layer (Streamlit)"
        App[app.py<br/>Main Entry Point]
        Home[Home.py<br/>Landing Page]
        Pred[Prediction.py<br/>Risk Assessment]
        EDA[EDA.py<br/>Data Analysis]
        FI[Feature_Importance.py<br/>Model Insights]
        MM[Model_Metrics.py<br/>Performance]
    end

    subgraph "Business Logic Layer"
        Utils[apps/utils.py<br/>Model Loading<br/>Predictions<br/>Visualizations]
        FE[src/feature_engineering.py<br/>process_apps()<br/>behaviorial_features()]
    end

    subgraph "Model Layer"
        M1[models/model_hybrid.pkl<br/>Traditional LightGBM]
        M2[models/first_lgbm_model.pkl<br/>Behavioral LightGBM]
        M3[models/model_ensemble_wrapper.pkl<br/>Ensemble Model]
    end

    subgraph "Data Layer"
        TrainData[Training Data<br/>smoke_engineered.csv<br/>uci_interface_test.csv]
        TestData[Test Data<br/>smoke_hybrid_features.csv]
        UserData[User Uploads<br/>CSV Files]
    end

    subgraph "Configuration"
        Env[Environment<br/>myenv/<br/>Python 3.13]
        Req[Dependencies<br/>requirements.txt]
    end

    App --> Home
    App --> Pred
    App --> EDA
    App --> FI
    App --> MM

    Pred --> Utils
    EDA --> Utils
    FI --> Utils
    MM --> Utils

    Utils --> FE
    Utils --> M1
    Utils --> M2
    Utils --> M3

    FE --> TrainData
    M1 --> TrainData
    M2 --> TrainData
    M3 --> TrainData

    UserData --> Pred
    TestData --> Pred

    Env --> App
    Req --> Env

    style App fill:#4fc3f7
    style Utils fill:#fff59d
    style FE fill:#aed581
    style M1 fill:#ffb74d
    style M2 fill:#f06292
    style M3 fill:#ba68c8
```

---

#### Detailed Component Architecture

##### 1. Frontend Components (Streamlit Pages)

```mermaid
graph LR
    subgraph "Navigation"
        Nav[Sidebar Navigation]
    end

    subgraph "Home Page"
        H1[System Overview]
        H2[Model Selection Guide]
        H3[Quick Start]
    end

    subgraph "Prediction Page"
        P1[Model Selector]
        P2[Input Mode Selector]
        P3[Manual Input Form]
        P4[CSV Upload]
        P5[Results Display]
        P6[Risk Visualization]
    end

    subgraph "EDA Page"
        E1[Data Upload]
        E2[Statistical Summary]
        E3[Distribution Plots]
        E4[Correlation Matrix]
        E5[Feature Analysis]
    end

    subgraph "Feature Importance"
        F1[Model Selector]
        F2[SHAP Values]
        F3[Feature Rankings]
        F4[Interactive Plots]
    end

    subgraph "Model Metrics"
        M1[Performance Metrics]
        M2[Confusion Matrix]
        M3[ROC Curve]
        M4[Precision-Recall]
    end

    Nav --> H1
    Nav --> P1
    Nav --> E1
    Nav --> F1
    Nav --> M1

    style Nav fill:#4fc3f7
    style P1 fill:#fff59d
    style E1 fill:#aed581
    style F1 fill:#ffb74d
    style M1 fill:#f06292
```

---

##### 2. Business Logic Components

```mermaid
graph TB
    subgraph "apps/utils.py - Core Functions"
        U1[load_model<br/>Load .pkl files]
        U2[get_available_models<br/>Scan models/ directory]
        U3[get_model_type<br/>Identify model category]
        U4[get_predictions<br/>Make predictions]
        U5[align_features<br/>Match model schema]
        U6[classify_risk<br/>Risk categorization]
        U7[plot_gauge<br/>Visualization]
    end

    subgraph "src/feature_engineering.py"
        FE1[process_apps<br/>Traditional Engineering<br/>11 → 487 features]
        FE2[behaviorial_features<br/>Behavioral Engineering<br/>23 → 31 features]
        FE3[Helper Functions<br/>Aggregations<br/>Calculations]
    end

    U4 --> U5
    U5 --> U1
    U4 --> FE1
    U4 --> FE2
    FE1 --> FE3
    FE2 --> FE3

    style U1 fill:#4fc3f7
    style U4 fill:#fff59d
    style FE1 fill:#aed581
    style FE2 fill:#f06292
```

---

##### 3. File Structure Tree

```
Loan Default Hybrid System/
│
├── app.py                          # Main Streamlit application
├── requirement.txt                 # Python dependencies
├── MODEL_ARCHITECTURE_FLOWCHART.md # This documentation
├── DATA_FLOW_EXPLANATION.md        # Data flow docs
│
├── myenv/                          # Virtual environment
│   ├── Scripts/                       # Python executables
│   └── Lib/                           # Installed packages
│
├── pages/                          # Streamlit pages
│   ├── Prediction.py                  # Main prediction interface
│   ├── EDA.py                         # Exploratory data analysis
│   ├── Feature_Importance.py          # Feature importance plots
│   └── Model_Metrics.py               # Model performance metrics
│
├── apps/                           # Business logic
│   └── utils.py                       # Core utility functions
│
├── src/                            # Source code
│   └── feature_engineering.py         # Feature engineering functions
│
├── models/                         # Trained models
│   ├── model_hybrid.pkl               # Traditional model (487 features)
│   ├── first_lgbm_model.pkl           # Behavioral model (31 features)
│   └── model_ensemble_wrapper.pkl     # Ensemble model (518 features)
│
└── data/ (optional)                # Training/test data
    ├── smoke_engineered.csv           # Traditional features
    ├── uci_interface_test.csv         # Behavioral features
    └── smoke_hybrid_features.csv      # Hybrid features
```

---

##### 4. Technology Stack

```mermaid
graph TB
    subgraph "Frontend"
        ST[Streamlit 1.x<br/>Web Framework]
        PL[Plotly<br/>Interactive Visualizations]
    end

    subgraph "Data Processing"
        PD[Pandas<br/>Data Manipulation]
        NP[NumPy<br/>Numerical Computing]
    end

    subgraph "Machine Learning"
        LGB[LightGBM<br/>Gradient Boosting]
        SK[Scikit-learn<br/>ML Utilities]
    end

    subgraph "Interpretability"
        SH[SHAP<br/>Model Explanations]
    end

    subgraph "Runtime"
        PY[Python 3.13<br/>Core Runtime]
        VE[Virtual Environment<br/>myenv]
    end

    ST --> PL
    ST --> PD
    PD --> NP
    LGB --> SK
    SK --> PD
    SH --> LGB

    PY --> ST
    PY --> PD
    PY --> LGB
    VE --> PY

    style ST fill:#4fc3f7
    style LGB fill:#ffb74d
    style PD fill:#aed581
    style PY fill:#f06292
```

**Key Dependencies:**

- **Streamlit**: Web application framework
- **LightGBM**: Gradient boosting model
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Plotly**: Interactive visualizations
- **SHAP**: Model interpretability
- **Scikit-learn**: ML utilities

---

##### 5. Request-Response Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Page as Prediction Page
    participant Utils as apps/utils.py
    participant FE as Feature Engineering
    participant Model as LightGBM Model

    User->>UI: Access Application
    UI->>Page: Navigate to Prediction

    Page->>Utils: get_available_models()
    Utils->>Page: Return [model1, model2, model3]
    Page->>User: Show Model Selection

    User->>Page: Select Model & Input Data
    Page->>Utils: get_model_type(model_name)
    Utils->>Page: Return 'traditional'/'behavioral'/'ensemble'

    alt Manual Input
        Page->>Page: Display Model-Specific Form
        User->>Page: Fill Form (11/23/34 features)
    else CSV Upload
        User->>Page: Upload CSV File
        Page->>Page: Load CSV with Pandas
    end

    Page->>FE: Apply Feature Engineering

    alt Traditional
        FE->>FE: process_apps(11 features)
        FE->>Page: Return 487 features
    else Behavioral
        FE->>FE: behaviorial_features(23 features)
        FE->>Page: Return 31 features
    else Ensemble
        FE->>FE: process_apps + behaviorial_features
        FE->>Page: Return 518 features
    end

    Page->>Utils: get_predictions(model, data)
    Utils->>Utils: load_model(model_path)
    Utils->>Utils: align_features(data, model)
    Utils->>Model: model.predict(X)
    Model->>Utils: Return probabilities
    Utils->>Page: Return [predictions, probabilities]

    Page->>Utils: classify_risk(probability)
    Utils->>Page: Return risk_level

    Page->>Utils: plot_gauge(probability)
    Utils->>Page: Return plotly_figure

    Page->>User: Display Results + Visualization
```

---

##### 6. Data Flow Through System

```mermaid
graph TB
    subgraph "Input Sources"
        I1[Manual Form Input]
        I2[CSV File Upload]
    end

    subgraph "Data Validation"
        V1[Check Required Fields]
        V2[Validate Data Types]
        V3[Check Value Ranges]
    end

    subgraph "Feature Engineering Pipeline"
        FE1{Model Type?}
        FE2[Traditional Pipeline<br/>11 → 487]
        FE3[Behavioral Pipeline<br/>23 → 31]
        FE4[Hybrid Pipeline<br/>34 → 518]
    end

    subgraph "Model Inference"
        M1[Load Model]
        M2[Align Features]
        M3[Predict]
        M4[Calculate Probabilities]
    end

    subgraph "Post-Processing"
        P1[Risk Classification]
        P2[Generate Visualizations]
        P3[Format Results]
    end

    subgraph "Output"
        O1[Display Results]
        O2[Download CSV]
        O3[Show Charts]
    end

    I1 --> V1
    I2 --> V1
    V1 --> V2
    V2 --> V3
    V3 --> FE1

    FE1 -->|Traditional| FE2
    FE1 -->|Behavioral| FE3
    FE1 -->|Ensemble| FE4

    FE2 --> M1
    FE3 --> M1
    FE4 --> M1

    M1 --> M2
    M2 --> M3
    M3 --> M4

    M4 --> P1
    P1 --> P2
    P2 --> P3

    P3 --> O1
    P3 --> O2
    P3 --> O3

    style I1 fill:#e1f5ff
    style FE1 fill:#fff59d
    style M3 fill:#ffb74d
    style O1 fill:#c8e6c9
```

---

##### 7. Model Loading & Caching Strategy

```mermaid
graph TB
    Start[User Selects Model] --> Cache{Model in<br/>Streamlit Cache?}

    Cache -->|Yes| LoadCache[Load from Cache]
    Cache -->|No| LoadDisk[Load from Disk]

    LoadDisk --> Unpickle[Unpickle .pkl file]
    Unpickle --> Store[Store in st.cache]
    Store --> Return[Return Model]

    LoadCache --> Return

    Return --> Ready[Model Ready<br/>for Inference]

    style Cache fill:#fff59d
    style Store fill:#aed581
    style Ready fill:#c8e6c9
```

**Caching Benefits:**

- Models loaded once per session
- Faster subsequent predictions
- Reduced memory overhead
- Better user experience

---

##### 8. Error Handling & Validation

```mermaid
graph TB
    Input[User Input] --> V1{Valid Format?}

    V1 -->|No| E1[Show Format Error]
    V1 -->|Yes| V2{Required Features?}

    V2 -->|No| E2[Show Missing Features]
    V2 -->|Yes| V3{Value Ranges Valid?}

    V3 -->|No| E3[Show Range Error]
    V3 -->|Yes| Process[Process Data]

    Process --> V4{Engineering Success?}
    V4 -->|No| E4[Show Engineering Error]
    V4 -->|Yes| Predict[Make Prediction]

    Predict --> V5{Prediction Success?}
    V5 -->|No| E5[Show Model Error]
    V5 -->|Yes| Success[Display Results]

    E1 --> End[Error Message to User]
    E2 --> End
    E3 --> End
    E4 --> End
    E5 --> End

    style V1 fill:#fff59d
    style Success fill:#c8e6c9
    style End fill:#ffccbc
```

---

#### 9. Deployment Architecture (Production Ready)

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
    end

    subgraph "Web Server"
        Streamlit[Streamlit Server<br/>Port 8501]
    end

    subgraph "Application Layer"
        App[Python Application<br/>app.py]
        Pages[Page Modules]
        Utils[Utility Functions]
    end

    subgraph "ML Layer"
        Models[Serialized Models<br/>.pkl files]
        FE[Feature Engineering<br/>Functions]
    end

    subgraph "Data Storage"
        Local[Local CSV Files]
        Upload[User Uploads<br/>Temp Storage]
    end

    Browser -->|HTTP| Streamlit
    Streamlit -->|WSGI| App
    App --> Pages
    Pages --> Utils
    Utils --> Models
    Utils --> FE
    FE --> Local
    Pages --> Upload

    style Browser fill:#e1f5ff
    style Streamlit fill:#4fc3f7
    style Models fill:#ffb74d
    style Local fill:#aed581
```

**Deployment Options:**

- **Local**: `streamlit run app.py`
- **Cloud**: Streamlit Community Cloud, Heroku, AWS, Azure
- **Docker**: Containerized deployment
- **Requirements**: Python 3.13, ~500MB models, 2GB RAM minimum

---

#### 10. Security & Performance Considerations

```mermaid
graph LR
    subgraph "Security"
        S1[Input Validation]
        S2[File Type Checking]
        S3[Size Limits]
        S4[No Data Persistence]
    end

    subgraph "Performance"
        P1[Model Caching]
        P2[Lazy Loading]
        P3[Batch Processing]
        P4[Memory Management]
    end

    subgraph "Reliability"
        R1[Error Handling]
        R2[Fallback Values]
        R3[Logging]
        R4[User Feedback]
    end

    style S1 fill:#ffccbc
    style P1 fill:#aed581
    style R1 fill:#fff59d
```

---

### Summary

This hybrid system provides three specialized models:

1. **Traditional Model**: Deep feature engineering from 11 base features → 487 features
2. **Behavioral Model**: Payment pattern analysis from 23 base features → 31 features
3. **Ensemble Model**: Combined approach using both feature sets → 518 features

Each model uses LightGBM for predictions and returns a probability (0-1) representing default risk, which is then classified into Low/Medium/High risk categories for actionable insights.

#### System Highlights:

- **Modular Architecture**: Clean separation of concerns (UI, Logic, Models, Data)
- **Scalable Design**: Easy to add new models or features
- **User-Friendly**: Streamlit provides intuitive interface
- **Production-Ready**: Error handling, caching, validation
- **Well-Documented**: Comprehensive flowcharts and architecture diagrams
