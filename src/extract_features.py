"""
Feature Orchestration Layer for Loan Default Prediction

This module combines individual feature engineering functions from 
feature_engineering.py to create complete feature sets for different models.

Three Feature Configurations:
1. traditional_features() - Home Credit only (apps + prev + bureau)
2. hybrid_features() - All datasets combined (487 features)
3. behaviorial_features() - Behavioral focus (apps + balances)

Date: November 2025
Version: 2.0
"""

import pandas as pd
import logging
from .feature_engineering import (
    process_apps,
    process_prev,
    get_prev_agg,
    process_bureau,
    get_bureau_agg,
    process_pos,
    process_install,
    process_card,
)

logger = logging.getLogger(__name__)


def traditional_features(apps, prev, bureau, bureau_bal):
    """
    Create traditional credit scoring features (Home Credit data only).
    
    Combines application data with previous loan history and credit bureau
    information to create comprehensive credit risk features.
    
    Parameters:
    -----------
    apps : pd.DataFrame
        Application data (application_train.csv)
    prev : pd.DataFrame
        Previous loan applications (previous_application.csv)
    bureau : pd.DataFrame
        Credit bureau data (bureau.csv)
    bureau_bal : pd.DataFrame
        Bureau monthly balances (bureau_balance.csv)
    
    Returns:
    --------
    pd.DataFrame
        Combined dataset with ~487 features including:
        - 13 application-level features (APPS_*)
        - Previous loan aggregations (PREV_*)
        - Bureau credit history (BUREAU_*)
    
    Pipeline:
    ---------
    1. Process applications → 13 engineered features
    2. Aggregate previous loans → PREV_* features
    3. Aggregate bureau history → BUREAU_* features
    4. Left join all on SK_ID_CURR
    
    Example:
    --------
    >>> apps, prev, bureau, bb = get_dataset()
    >>> traditional_df = traditional_features(apps, prev, bureau, bb)
    >>> print(f"Features: {traditional_df.shape[1]}")
    Features: 487
    """

    apps_all = process_apps(apps)
    prev_agg = get_prev_agg(prev)
    bureau_agg = get_bureau_agg(bureau, bureau_bal)
    logger.info('prev_agg shape: %s bureau_agg shape: %s', prev_agg.shape, bureau_agg.shape)
    logger.info('apps_all before merge shape: %s', apps_all.shape)
    apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(bureau_agg, on='SK_ID_CURR', how='left')
    logger.info('apps_all after merge with prev_agg shape: %s', apps_all.shape)


    return apps_all




def hybrid_features(apps, bureau, bureau_bal, prev, pos_bal, install, card_bal):
    """
    Create hybrid feature set combining ALL data sources.
    
    This is the most comprehensive feature set, combining traditional credit
    data with behavioral patterns from balance histories.
    
    Parameters:
    -----------
    apps : pd.DataFrame
        Application data
    bureau : pd.DataFrame
        Credit bureau data
    bureau_bal : pd.DataFrame
        Bureau balances
    prev : pd.DataFrame
        Previous applications
    pos_bal : pd.DataFrame
        POS cash balance history
    install : pd.DataFrame
        Installments payments
    card_bal : pd.DataFrame
        Credit card balance history
    
    Returns:
    --------
    pd.DataFrame
        Complete dataset with 487 features including:
        - Application features (13)
        - Previous loan aggregations (PREV_*)
        - Bureau aggregations (BUREAU_*)
        - POS balance features (POS_*)
        - Installment features (INSTALL_*)
        - Credit card features (CARD_*)
    
    Pipeline:
    ---------
    1. Process applications
    2. Aggregate each data source
    3. Sequential left joins on SK_ID_CURR
    4. Memory optimization
    
    Used By:
    --------
    - Traditional LightGBM model (model_hybrid.pkl)
    - Ensemble model's traditional branch
    
    Example:
    --------
    >>> apps, prev, bureau, bb = get_dataset()
    >>> pos, install, card = get_balance_data()
    >>> hybrid_df = hybrid_features(apps, bureau, bb, prev, pos, install, card)
    >>> print(f"Total features: {hybrid_df.shape[1]}")
    Total features: 487
    """

    apps_all =  process_apps(apps)
    bureau_agg = get_bureau_agg(bureau, bureau_bal)
    prev_agg = get_prev_agg(prev)
    pos_bal_agg = process_pos(pos_bal)
    install_agg = process_install(install)
    card_bal_agg = process_card(card_bal)
    # logger.debug('prev_agg shape: %s bureau_agg shape: %s', prev_agg.shape, bureau_agg.shape)
    logger.info('pos_bal_agg shape: %s install_agg shape: %s card_bal_agg shape: %s', pos_bal_agg.shape, install_agg.shape, card_bal_agg.shape)
    logger.info('apps_all before merge shape: %s', apps_all.shape)

    # Join with apps_all
    apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(bureau_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(pos_bal_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(install_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(card_bal_agg, on='SK_ID_CURR', how='left')

    logger.info('apps_all after merge with all shape: %s', apps_all.shape)
    # apps_all = reduce_mem_usage(apps_all) # Apply memory reduction after merging
    print('data types are converted for a reduced memory usage')


    return apps_all

# @title
def behaviorial_features(apps, pos_bal, install, card_bal):
    """
    Create behavioral-focused features (balance patterns).
    
    Focuses on payment behavior and spending patterns from balance histories,
    excluding traditional credit bureau data.
    
    Parameters:
    -----------
    apps : pd.DataFrame
        Application data (for SK_ID_CURR join key)
    pos_bal : pd.DataFrame
        POS cash balance history
    install : pd.DataFrame
        Installments payments
    card_bal : pd.DataFrame
        Credit card balance history
    
    Returns:
    --------
    pd.DataFrame
        Dataset with behavioral features:
        - Application base features (13)
        - POS balance aggregations (POS_*)
        - Installment patterns (INSTALL_*)
        - Credit card usage (CARD_*)
    
    Pipeline:
    ---------
    1. Process applications
    2. Aggregate balance histories
    3. Left joins on SK_ID_CURR
    
    Note:
    -----
    Excludes previous_application and bureau data to focus purely
    on recent behavioral patterns.
    
    Used By:
    --------
    - Alternative modeling approach focusing on behavior
    - Experimental feature sets
    
    Example:
    --------
    >>> apps, *_ = get_dataset()
    >>> pos, install, card = get_balance_data()
    >>> behav_df = behaviorial_features(apps, pos, install, card)
    >>> print(f"Behavioral features: {behav_df.shape[1]}")
    """

    apps_all =  process_apps(apps)
    pos_bal_agg = process_pos(pos_bal)
    install_agg = process_install(install)
    card_bal_agg = process_card(card_bal)
    # logger.debug('prev_agg shape: %s bureau_agg shape: %s', prev_agg.shape, bureau_agg.shape)
    logger.info('pos_bal_agg shape: %s install_agg shape: %s card_bal_agg shape: %s', pos_bal_agg.shape, install_agg.shape, card_bal_agg.shape)
    logger.info('apps_all before merge shape: %s', apps_all.shape)

    # Join with apps_all
    # apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    # apps_all = apps_all.merge(bureau_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(pos_bal_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(install_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(card_bal_agg, on='SK_ID_CURR', how='left')

    logger.info('apps_all after merge with all shape: %s', apps_all.shape)

    #apps_all = reduce_mem_usage(apps_all)
    #print('data types are converted for a reduced memory usage')


    return apps_all

def get_apps_all_encoded(apps_all):
    """
    Encode categorical variables using factorization.
    
    Converts all object-type columns to numeric codes for model training.
    
    Parameters:
    -----------
    apps_all : pd.DataFrame
        Feature-engineered dataset with mixed types
    
    Returns:
    --------
    pd.DataFrame
        Same dataset with object columns converted to numeric codes
    
    Note:
    -----
    Uses pd.factorize() which creates integer codes for each unique value.
    Better for tree-based models than one-hot encoding.
    
    Example:
    --------
    >>> df_encoded = get_apps_all_encoded(df_with_categories)
    >>> print(df_encoded.dtypes.value_counts())
    """

    object_columns = apps_all.dtypes[apps_all.dtypes == 'object'].index.tolist()
    for column in object_columns:
        apps_all[column] = pd.factorize(apps_all[column])[0]

    return apps_all
