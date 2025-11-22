"""
Script to create hybrid features by combining traditional Home Credit features
with behavioral UCI Credit Card features for a comprehensive hybrid model.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from feature_engineering import behaviorial_features, process_apps

def load_datasets():
    """Load both smoke_engineered and uci_interface_test datasets"""
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load Home Credit smoke data (has traditional features)
    smoke_df = pd.read_csv(data_dir / "smoke_engineered.csv")
    print(f"Loaded smoke_engineered.csv: {smoke_df.shape}")
    
    # Load UCI Credit Card data (has behavioral features)
    uci_df = pd.read_csv(data_dir / "uci_interface_test.csv")
    print(f"Loaded uci_interface_test.csv: {uci_df.shape}")
    
    return smoke_df, uci_df

def simulate_behavioral_features_for_smoke(smoke_df):
    """
    Simulate behavioral features for smoke dataset users
    Based on their existing credit behavior and demographics
    """
    print("\nSimulating behavioral features for Home Credit users...")
    
    behavioral_sim = pd.DataFrame(index=smoke_df.index)
    
    # Simulate UCI original features first
    if 'AMT_CREDIT' in smoke_df.columns:
        behavioral_sim['LIMIT_BAL'] = smoke_df['AMT_CREDIT'] * np.random.uniform(0.5, 1.5, len(smoke_df))
    else:
        behavioral_sim['LIMIT_BAL'] = np.random.uniform(10000, 500000, len(smoke_df))
    
    # SEX: 1=male, 2=female
    if 'CODE_GENDER' in smoke_df.columns:
        behavioral_sim['SEX'] = smoke_df['CODE_GENDER'].map({'M': 1, 'F': 2}).fillna(1)
    else:
        behavioral_sim['SEX'] = np.random.choice([1, 2], len(smoke_df))
    
    # EDUCATION: 1=grad, 2=university, 3=high school, 4=others
    if 'NAME_EDUCATION_TYPE' in smoke_df.columns:
        edu_map = {
            'Higher education': 2,
            'Secondary / secondary special': 3,
            'Incomplete higher': 2,
            'Lower secondary': 4,
            'Academic degree': 1
        }
        behavioral_sim['EDUCATION'] = smoke_df['NAME_EDUCATION_TYPE'].map(edu_map).fillna(3)
    else:
        behavioral_sim['EDUCATION'] = np.random.choice([1, 2, 3, 4], len(smoke_df), p=[0.1, 0.4, 0.3, 0.2])
    
    # MARRIAGE: 1=married, 2=single, 3=others
    if 'NAME_FAMILY_STATUS' in smoke_df.columns:
        marriage_map = {
            'Married': 1,
            'Single / not married': 2,
            'Civil marriage': 1,
            'Widow': 3,
            'Separated': 3
        }
        behavioral_sim['MARRIAGE'] = smoke_df['NAME_FAMILY_STATUS'].map(marriage_map).fillna(2)
    else:
        behavioral_sim['MARRIAGE'] = np.random.choice([1, 2, 3], len(smoke_df), p=[0.5, 0.35, 0.15])
    
    # AGE
    if 'DAYS_BIRTH' in smoke_df.columns:
        behavioral_sim['AGE'] = (-smoke_df['DAYS_BIRTH'] / 365).astype(int).clip(21, 70)
    else:
        behavioral_sim['AGE'] = np.random.randint(21, 70, len(smoke_df))
    
    # PAY_0 to PAY_6 (payment status: -1=pay duly, 1=delay 1 month, 2=delay 2 months, etc.)
    for i in [0, 2, 3, 4, 5, 6]:
        behavioral_sim[f'PAY_{i}'] = np.random.choice([-1, 0, 1, 2], len(smoke_df), p=[0.6, 0.2, 0.15, 0.05])
    
    # Use existing features as basis for simulation
    # Simulate BILL_AMT columns based on AMT_CREDIT and income patterns
    if 'AMT_CREDIT' in smoke_df.columns and 'AMT_INCOME_TOTAL' in smoke_df.columns:
        credit_utilization_base = smoke_df['AMT_CREDIT'] / (smoke_df['AMT_INCOME_TOTAL'] + 1)
        
        # Simulate 6 months of bill amounts with some randomness
        for i in range(1, 7):
            noise = np.random.normal(1, 0.2, len(smoke_df))
            behavioral_sim[f'BILL_AMT{i}'] = (credit_utilization_base * smoke_df['AMT_INCOME_TOTAL'] * 0.3 * noise).clip(0)
    else:
        # Fallback: generate random values
        for i in range(1, 7):
            behavioral_sim[f'BILL_AMT{i}'] = np.random.uniform(1000, 100000, len(smoke_df))
    
    # Simulate PAY_AMT columns based on income and credit behavior
    if 'AMT_ANNUITY' in smoke_df.columns:
        payment_base = smoke_df['AMT_ANNUITY'].fillna(smoke_df['AMT_ANNUITY'].median())
        
        for i in range(1, 7):
            noise = np.random.normal(1, 0.25, len(smoke_df))
            behavioral_sim[f'PAY_AMT{i}'] = (payment_base * noise).clip(0)
    else:
        for i in range(1, 7):
            behavioral_sim[f'PAY_AMT{i}'] = np.random.uniform(500, 50000, len(smoke_df))
    
    # Apply behavioral feature engineering
    behavioral_sim = behaviorial_features(behavioral_sim)
    
    print(f"Created {behavioral_sim.shape[1]} simulated behavioral features")
    
    return behavioral_sim

def simulate_traditional_features_for_uci(uci_df):
    """
    Simulate traditional Home Credit features for UCI users
    Based on their behavioral patterns and credit card usage
    """
    print("\nSimulating traditional features for UCI Credit Card users...")
    
    traditional_sim = pd.DataFrame(index=uci_df.index)
    
    # Map UCI features to Home Credit features where possible
    
    # AGE mapping
    if 'AGE' in uci_df.columns:
        traditional_sim['DAYS_BIRTH'] = -(uci_df['AGE'] * 365)
    else:
        traditional_sim['DAYS_BIRTH'] = np.random.uniform(-25000, -7000, len(uci_df))
    
    # Simulate income based on credit limit
    if 'LIMIT_BAL' in uci_df.columns:
        traditional_sim['AMT_INCOME_TOTAL'] = uci_df['LIMIT_BAL'] * np.random.uniform(2, 6, len(uci_df))
    else:
        traditional_sim['AMT_INCOME_TOTAL'] = np.random.uniform(50000, 500000, len(uci_df))
    
    # Simulate credit amount based on limit balance
    if 'LIMIT_BAL' in uci_df.columns:
        traditional_sim['AMT_CREDIT'] = uci_df['LIMIT_BAL'] * np.random.uniform(0.3, 0.9, len(uci_df))
    else:
        traditional_sim['AMT_CREDIT'] = np.random.uniform(100000, 1000000, len(uci_df))
    
    # Simulate annuity (monthly payment)
    traditional_sim['AMT_ANNUITY'] = traditional_sim['AMT_CREDIT'] / np.random.uniform(12, 60, len(uci_df))
    
    # Simulate goods price
    traditional_sim['AMT_GOODS_PRICE'] = traditional_sim['AMT_CREDIT'] * np.random.uniform(0.9, 1.1, len(uci_df))
    
    # External source scores (credit scores) - simulate based on payment history
    traditional_sim['EXT_SOURCE_1'] = np.random.uniform(0.2, 0.8, len(uci_df))
    traditional_sim['EXT_SOURCE_2'] = np.random.uniform(0.2, 0.8, len(uci_df))
    traditional_sim['EXT_SOURCE_3'] = np.random.uniform(0.2, 0.8, len(uci_df))
    
    # Days employed - simulate based on age
    traditional_sim['DAYS_EMPLOYED'] = traditional_sim['DAYS_BIRTH'] * np.random.uniform(0.4, 0.7, len(uci_df))
    
    # Family members
    traditional_sim['CNT_FAM_MEMBERS'] = np.random.choice([1, 2, 3, 4, 5], len(uci_df), p=[0.3, 0.35, 0.2, 0.1, 0.05])
    
    # Car age
    traditional_sim['OWN_CAR_AGE'] = np.random.choice([np.nan, 5, 10, 15, 20], len(uci_df), p=[0.5, 0.2, 0.15, 0.1, 0.05])
    
    # Apply traditional feature engineering
    traditional_sim_df = process_apps(traditional_sim)
    
    print(f"Created {traditional_sim_df.shape[1]} simulated traditional features")
    
    return traditional_sim_df

def create_hybrid_dataset(smoke_df, uci_df):
    """
    Create hybrid datasets by adding simulated features to both datasets
    """
    print("\n" + "="*60)
    print("Creating Hybrid Feature Sets")
    print("="*60)
    
    # Preserve TARGET column if it exists
    has_target = 'TARGET' in smoke_df.columns
    if has_target:
        target_col = smoke_df['TARGET'].copy()
        print(f"Preserving TARGET column ({target_col.notnull().sum()} non-null values)")
    
    # For smoke dataset: add simulated behavioral features
    smoke_behavioral = simulate_behavioral_features_for_smoke(smoke_df)
    smoke_hybrid = pd.concat([smoke_df, smoke_behavioral], axis=1)
    
    # Restore TARGET if it was present
    if has_target:
        smoke_hybrid['TARGET'] = target_col
    
    # For UCI dataset: add simulated traditional features
    uci_traditional = simulate_traditional_features_for_uci(uci_df)
    uci_hybrid = pd.concat([uci_df, uci_traditional], axis=1)
    
    print(f"\nSmoke hybrid shape: {smoke_hybrid.shape}")
    print(f"UCI hybrid shape: {uci_hybrid.shape}")
    
    return smoke_hybrid, uci_hybrid

def get_hybrid_predictions(model1, model2, X1, X2):
    """
    Get predictions from both models and create ensemble prediction
    
    Args:
        model1: First model (e.g., model_hybrid - traditional)
        model2: Second model (e.g., first_lgbm_model - behavioral)
        X1: Features for model1
        X2: Features for model2
    
    Returns:
        Combined predictions and probabilities
    """
    print("\nGenerating hybrid predictions...")
    
    # Get predictions from both models
    pred1 = model1.predict(X1)
    proba1 = model1.predict_proba(X1)[:, 1]
    
    pred2 = model2.predict(X2)
    proba2 = model2.predict_proba(X2)[:, 1]
    
    # Ensemble strategy: weighted average of probabilities
    # You can adjust weights based on model performance
    weight1 = 0.6  # Weight for traditional model
    weight2 = 0.4  # Weight for behavioral model
    
    ensemble_proba = (weight1 * proba1) + (weight2 * proba2)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    print(f"Model 1 predictions - Mean probability: {proba1.mean():.4f}")
    print(f"Model 2 predictions - Mean probability: {proba2.mean():.4f}")
    print(f"Ensemble predictions - Mean probability: {ensemble_proba.mean():.4f}")
    
    return ensemble_pred, ensemble_proba, pred1, proba1, pred2, proba2

def save_hybrid_datasets(smoke_hybrid, uci_hybrid, output_dir):
    """Save the hybrid datasets for future use"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    smoke_hybrid.to_csv(output_path / "smoke_hybrid_features.csv", index=False)
    uci_hybrid.to_csv(output_path / "uci_hybrid_features.csv", index=False)
    
    print(f"\nSaved hybrid datasets to {output_path}")

def main():
    """Main execution function"""
    print("="*60)
    print("HYBRID FEATURE CREATION FOR ENSEMBLE MODEL")
    print("="*60)
    
    # Load datasets
    smoke_df, uci_df = load_datasets()
    
    # Create hybrid feature sets
    smoke_hybrid, uci_hybrid = create_hybrid_dataset(smoke_df, uci_df)
    
    # Save hybrid datasets
    output_dir = Path(__file__).parent.parent / "data"
    save_hybrid_datasets(smoke_hybrid, uci_hybrid, output_dir)
    
    print("\n" + "="*60)
    print("HYBRID FEATURE CREATION COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use smoke_hybrid_features.csv with both model_hybrid.pkl and first_lgbm_model.pkl")
    print("2. Create ensemble predictions by combining outputs from both models")
    print("3. Train a meta-learner (stacking) if needed for optimal weight combination")

if __name__ == "__main__":
    main()
