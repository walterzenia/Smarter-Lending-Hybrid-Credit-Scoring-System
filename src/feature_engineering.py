"""
Feature Engineering Pipeline for Loan Default Prediction

This module implements comprehensive feature engineering for both traditional
credit scoring (Home Credit data) and behavioral analysis (UCI Credit Card data).

Pipeline Structure:
1. Traditional Features: Credit history, demographics, external scores
2. Behavioral Features: Payment patterns, spending behavior, utilization
3. Hybrid Features: Combined features for ensemble model

Date: November 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional

# Ensure we operate on copies to avoid SettingWithCopyWarning when callers pass slices
def _ensure_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Create a copy of the DataFrame to prevent mutation of original data."""
    return df.copy()


def process_apps(apps: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for Home Credit application data (Traditional Model).
    
    Creates 13 new features from raw application data including:
    - Credit score aggregations (mean, std of external sources)
    - Financial ratios (annuity/credit, goods/credit, etc.)
    - Income-based ratios (credit/income, goods/income, etc.)
    - Temporal ratios (employment/age, income/employment, etc.)
    
    Parameters:
    -----------
    apps : pd.DataFrame
        Raw application data with columns:
        - EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3: External credit scores
        - AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE: Loan amounts
        - AMT_INCOME_TOTAL: Annual income
        - CNT_FAM_MEMBERS: Family size
        - DAYS_BIRTH, DAYS_EMPLOYED: Age and employment indicators
        - OWN_CAR_AGE: Vehicle age
    
    Returns:
    --------
    pd.DataFrame
        Original data + 13 engineered features with 'APPS_' prefix
        
    Features Created:
    -----------------
    - APPS_EXT_SOURCE_MEAN: Average of external credit scores
    - APPS_EXT_SOURCE_STD: Variability in credit scores
    - APPS_ANNUITY_CREDIT_RATIO: Monthly payment / Total loan
    - APPS_GOODS_CREDIT_RATIO: Item price / Loan amount
    - APPS_ANNUITY_INCOME_RATIO: Payment burden relative to income
    - APPS_CREDIT_INCOME_RATIO: Loan size relative to income
    - APPS_GOODS_INCOME_RATIO: Purchase price / Annual income
    - APPS_CNT_FAM_INCOME_RATIO: Income per family member
    - APPS_EMPLOYED_BIRTH_RATIO: Employment length / Age
    - APPS_INCOME_EMPLOYED_RATIO: Income / Employment duration
    - APPS_INCOME_BIRTH_RATIO: Income / Age
    - APPS_CAR_BIRTH_RATIO: Car age / Person age
    - APPS_CAR_EMPLOYED_RATIO: Car age / Employment length
    
    Note:
    -----
    Infinite values from division are replaced with NaN.
    Original DataFrame is not modified (creates copy internally).
    
    Example:
    --------
    >>> apps_engineered = process_apps(raw_applications)
    >>> print(apps_engineered['APPS_CREDIT_INCOME_RATIO'].describe())
    """
    apps = _ensure_copy(apps)
    # 1. Deal with missing values in EXT_SOURCE columns
    # EXT_SOURCE
    apps['APPS_EXT_SOURCE_MEAN'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    apps['APPS_EXT_SOURCE_STD'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps['APPS_EXT_SOURCE_STD'].fillna(apps['APPS_EXT_SOURCE_STD'].mean())
    
    # 2. Ratios
    # AMT_CREDIT
    apps['APPS_ANNUITY_CREDIT_RATIO'] = apps['AMT_ANNUITY']/apps['AMT_CREDIT']
    apps['APPS_GOODS_CREDIT_RATIO'] = apps['AMT_GOODS_PRICE']/apps['AMT_CREDIT']

    # AMT_INCOME_TOTAL
    apps['APPS_ANNUITY_INCOME_RATIO'] = apps['AMT_ANNUITY']/apps['AMT_INCOME_TOTAL']
    apps['APPS_CREDIT_INCOME_RATIO'] = apps['AMT_CREDIT']/apps['AMT_INCOME_TOTAL']
    apps['APPS_GOODS_INCOME_RATIO'] = apps['AMT_GOODS_PRICE']/apps['AMT_INCOME_TOTAL']
    apps['APPS_CNT_FAM_INCOME_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['CNT_FAM_MEMBERS']

    # DAYS_BIRTH, DAYS_EMPLOYED
    apps['APPS_EMPLOYED_BIRTH_RATIO'] = apps['DAYS_EMPLOYED']/apps['DAYS_BIRTH']
    apps['APPS_INCOME_EMPLOYED_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['DAYS_EMPLOYED']
    apps['APPS_INCOME_BIRTH_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['DAYS_BIRTH']
    apps['APPS_CAR_BIRTH_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_BIRTH']
    apps['APPS_CAR_EMPLOYED_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_EMPLOYED']

    # Replace infinite values with NaN after ratio calculations
    apps.replace([np.inf, -np.inf], np.nan, inplace=True)

    return apps

def process_prev(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for previous loan application data.
    
    Creates 8 new features analyzing previous loan history including:
    - Credit differences and ratios
    - Timeline analysis (payment delays, due dates)
    - Interest rate estimation
    
    Parameters:
    -----------
    prev : pd.DataFrame
        Previous application data with columns:
        - AMT_APPLICATION, AMT_CREDIT, AMT_GOODS_PRICE: Loan amounts
        - AMT_ANNUITY: Monthly payment
        - CNT_PAYMENT: Number of payments
        - DAYS_DECISION: Days before decision
        - DAYS_FIRST_DRAWING, DAYS_FIRST_DUE: Payment timing
        - DAYS_LAST_DUE_1ST_VERSION, DAYS_LAST_DUE: Due date changes
        - DAYS_TERMINATION: Loan termination date
    
    Returns:
    --------
    pd.DataFrame
        Original data + 8 engineered features with 'PREV_' prefix
        
    Features Created:
    -----------------
    - PREV_CREDIT_DIFF: Requested vs approved loan amount difference
    - PREV_GOODS_DIFF: Requested vs goods price difference
    - PREV_CREDIT_APPL_RATIO: Approval rate (credit/application)
    - PREV_GOODS_APPL_RATIO: Goods price / Application amount
    - PREV_DAYS_LAST_DUE_DIFF: Change in due date over time
    - PREV_INTERESTS_RATE: Estimated interest rate
    
    Data Cleaning:
    --------------
    - Replaces sentinel value 365243 with NaN in date columns
    - Handles division by zero in interest rate calculation
    - Replaces infinite values with NaN
    
    Example:
    --------
    >>> prev_engineered = process_prev(previous_applications)
    >>> print(prev_engineered['PREV_CREDIT_APPL_RATIO'].mean())
    """
    prev = _ensure_copy(prev)
    prev['PREV_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
    prev['PREV_CREDIT_APPL_RATIO'] = prev['AMT_CREDIT']/prev['AMT_APPLICATION']
    prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE']/prev['AMT_APPLICATION']

    # Avoid chained assignment with inplace=True
    prev['DAYS_FIRST_DRAWING'] = prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan)
    prev['DAYS_FIRST_DUE'] = prev['DAYS_FIRST_DUE'].replace(365243, np.nan)
    prev['DAYS_LAST_DUE_1ST_VERSION'] = prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan)
    prev['DAYS_LAST_DUE'] = prev['DAYS_LAST_DUE'].replace(365243, np.nan)
    prev['DAYS_TERMINATION'] = prev['DAYS_TERMINATION'].replace(365243, np.nan)

    prev['PREV_DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']

    all_pay = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    # Handle potential division by zero before calculating the ratio
    prev['PREV_INTERESTS_RATE'] = (all_pay/prev['AMT_CREDIT'].replace(0, np.nan) - 1)/prev['CNT_PAYMENT'].replace(0, np.nan)

    # Replace infinite values with NaN after ratio calculation
    prev.replace([np.inf, -np.inf], np.nan, inplace=True)

    return prev

def get_prev_amt_agg(prev):

    agg_dict = {
      'SK_ID_CURR':['count'],
      'AMT_CREDIT':['mean', 'max', 'sum'],
      'AMT_ANNUITY':['mean', 'max', 'sum'],
      'AMT_APPLICATION':['mean', 'max', 'sum'],
      'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
      'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
      'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
      'DAYS_DECISION': ['min', 'max', 'mean'],
      'CNT_PAYMENT': ['mean', 'sum'],

      'PREV_CREDIT_DIFF':['mean', 'max', 'sum'],
      'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
      'PREV_GOODS_DIFF':['mean', 'max', 'sum'],
      'PREV_GOODS_APPL_RATIO':['mean', 'max'],
      'PREV_DAYS_LAST_DUE_DIFF':['mean', 'max', 'sum'],
      'PREV_INTERESTS_RATE':['mean', 'max']
    }

    prev_group = prev.groupby('SK_ID_CURR')
    prev_amt_agg = prev_group.agg(agg_dict)
    prev_amt_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]

    return prev_amt_agg

def get_prev_refused_appr_agg(prev):

    prev_refused_appr_group = prev[prev['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])].groupby([ 'SK_ID_CURR', 'NAME_CONTRACT_STATUS'])
    prev_refused_appr_agg = prev_refused_appr_group['SK_ID_CURR'].count().unstack()
    prev_refused_appr_agg.columns = ['PREV_APPROVED_COUNT', 'PREV_REFUSED_COUNT' ]
    prev_refused_appr_agg = prev_refused_appr_agg.fillna(0)
    return prev_refused_appr_agg


def get_prev_days365_agg(prev):

    cond_days365 = prev['DAYS_DECISION'] > -365
    prev_days365_group = prev[cond_days365].groupby('SK_ID_CURR')
    agg_dict = {
      'SK_ID_CURR':['count'],
      'AMT_CREDIT':['mean', 'max', 'sum'],
      'AMT_ANNUITY':['mean', 'max', 'sum'],
      'AMT_APPLICATION':['mean', 'max', 'sum'],
      'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
      'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
      'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
      'DAYS_DECISION': ['min', 'max', 'mean'],
      'CNT_PAYMENT': ['mean', 'sum'],

      'PREV_CREDIT_DIFF':['mean', 'max', 'sum'],
      'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
      'PREV_GOODS_DIFF':['mean', 'max', 'sum'],
      'PREV_GOODS_APPL_RATIO':['mean', 'max'],
      'PREV_DAYS_LAST_DUE_DIFF':['mean', 'max', 'sum'],
      'PREV_INTERESTS_RATE':['mean', 'max']
    }

    prev_days365_agg = prev_days365_group.agg(agg_dict)

    # multi index
    prev_days365_agg.columns = ["PREV_D365_"+ "_".join(x).upper() for x in prev_days365_agg.columns.ravel()]

    return prev_days365_agg

def get_prev_agg(prev):

    prev = process_prev(prev)
    prev_amt_agg = get_prev_amt_agg(prev)
    prev_refused_appr_agg = get_prev_refused_appr_agg(prev)
    prev_days365_agg = get_prev_days365_agg(prev)

    # prev_amt_agg
    prev_agg = prev_amt_agg.merge(prev_refused_appr_agg, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_days365_agg, on='SK_ID_CURR', how='left')
    # SK_ID_CURR APPROVED_COUNT REFUSED_COUNT
    prev_agg['PREV_REFUSED_RATIO'] = prev_agg['PREV_REFUSED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT']
    prev_agg['PREV_APPROVED_RATIO'] = prev_agg['PREV_APPROVED_COUNT']/prev_agg['PREV_SK_ID_CURR_COUNT']
    # 'PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT' drop
    prev_agg = prev_agg.drop(['PREV_REFUSED_COUNT', 'PREV_APPROVED_COUNT'], axis=1)

    return prev_agg

# Function for Bureau dataset feature engineering
def process_bureau(bureau: pd.DataFrame) -> pd.DataFrame:

    bureau = _ensure_copy(bureau)
    bureau['BUREAU_ENDDATE_FACT_DIFF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_FACT_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_ENDDATE_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['BUREAU_CREDIT_DEBT_RATIO']=bureau['AMT_CREDIT_SUM_DEBT']/bureau['AMT_CREDIT_SUM']
    bureau['BUREAU_CREDIT_DEBT_DIFF'] = bureau['AMT_CREDIT_SUM_DEBT'] - bureau['AMT_CREDIT_SUM']

    bureau['BUREAU_IS_DPD'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)
    bureau['BUREAU_IS_DPD_OVER120'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x >120 else 0)

    return bureau

def get_bureau_day_amt_agg(bureau):

    bureau_agg_dict = {
    'SK_ID_BUREAU':['count'],
    'DAYS_CREDIT':['min', 'max', 'mean'],
    'CREDIT_DAY_OVERDUE':['min', 'max', 'mean'],
    'DAYS_CREDIT_ENDDATE':['min', 'max', 'mean'],
    'DAYS_ENDDATE_FACT':['min', 'max', 'mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
    'AMT_ANNUITY': ['max', 'mean', 'sum'],

    'BUREAU_ENDDATE_FACT_DIFF':['min', 'max', 'mean'],
    'BUREAU_CREDIT_FACT_DIFF':['min', 'max', 'mean'],
    'BUREAU_CREDIT_ENDDATE_DIFF':['min', 'max', 'mean'],
    'BUREAU_CREDIT_DEBT_RATIO':['min', 'max', 'mean'],
    'BUREAU_CREDIT_DEBT_DIFF':['min', 'max', 'mean'],
    'BUREAU_IS_DPD':['mean', 'sum'],
    'BUREAU_IS_DPD_OVER120':['mean', 'sum']
    }

    bureau_grp = bureau.groupby('SK_ID_CURR')
    bureau_day_amt_agg = bureau_grp.agg(bureau_agg_dict)
    bureau_day_amt_agg.columns = ['BUREAU_'+('_').join(column).upper() for column in bureau_day_amt_agg.columns.ravel()]
    # SK_ID_CURR reset_index()
    bureau_day_amt_agg = bureau_day_amt_agg.reset_index()
    #print('bureau_day_amt_agg shape:', bureau_day_amt_agg.shape)
    return bureau_day_amt_agg

def get_bureau_active_agg(bureau):

    cond_active = bureau['CREDIT_ACTIVE'] == 'Active'
    bureau_active_grp = bureau[cond_active].groupby(['SK_ID_CURR'])
    bureau_agg_dict = {
      'SK_ID_BUREAU':['count'],
      'DAYS_CREDIT':['min', 'max', 'mean'],
      'CREDIT_DAY_OVERDUE':['min', 'max', 'mean'],
      'DAYS_CREDIT_ENDDATE':['min', 'max', 'mean'],
      'DAYS_ENDDATE_FACT':['min', 'max', 'mean'],
      'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
      'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
      'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
      'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
      'AMT_ANNUITY': ['max', 'mean', 'sum'],

      'BUREAU_ENDDATE_FACT_DIFF':['min', 'max', 'mean'],
      'BUREAU_CREDIT_FACT_DIFF':['min', 'max', 'mean'],
      'BUREAU_CREDIT_ENDDATE_DIFF':['min', 'max', 'mean'],
      'BUREAU_CREDIT_DEBT_RATIO':['min', 'max', 'mean'],
      'BUREAU_CREDIT_DEBT_DIFF':['min', 'max', 'mean'],
      'BUREAU_IS_DPD':['mean', 'sum'],
      'BUREAU_IS_DPD_OVER120':['mean', 'sum']
      }

    bureau_active_agg = bureau_active_grp.agg(bureau_agg_dict)
    bureau_active_agg.columns = ['BUREAU_ACT_'+('_').join(column).upper() for column in bureau_active_agg.columns.ravel()]
    bureau_active_agg = bureau_active_agg.reset_index()

    return bureau_active_agg


def get_bureau_bal_agg(bureau, bureau_bal):

    bureau_bal = bureau_bal.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], on='SK_ID_BUREAU', how='left')
    bureau_bal['BUREAU_BAL_IS_DPD'] = bureau_bal['STATUS'].apply(lambda x: 1 if x in['1','2','3','4','5']  else 0)
    bureau_bal['BUREAU_BAL_IS_DPD_OVER120'] = bureau_bal['STATUS'].apply(lambda x: 1 if x =='5'  else 0)
    bureau_bal_grp = bureau_bal.groupby('SK_ID_CURR')

    bureau_bal_agg_dict = {
        'SK_ID_CURR':['count'],
        'MONTHS_BALANCE':['min', 'max', 'mean'],
        'BUREAU_BAL_IS_DPD':['mean', 'sum'],
        'BUREAU_BAL_IS_DPD_OVER120':['mean', 'sum']
    }

    bureau_bal_agg = bureau_bal_grp.agg(bureau_bal_agg_dict)
    bureau_bal_agg.columns = [ 'BUREAU_BAL_'+('_').join(column).upper() for column in bureau_bal_agg.columns.ravel() ]

    bureau_bal_agg = bureau_bal_agg.reset_index()
    return bureau_bal_agg

def get_bureau_agg(bureau, bureau_bal):

    bureau = process_bureau(bureau)
    bureau_day_amt_agg = get_bureau_day_amt_agg(bureau)
    bureau_active_agg = get_bureau_active_agg(bureau)
    bureau_bal_agg = get_bureau_bal_agg(bureau, bureau_bal)

    bureau_agg = bureau_day_amt_agg.merge(bureau_active_agg, on='SK_ID_CURR', how='left')
    bureau_agg['BUREAU_ACT_IS_DPD_RATIO'] = bureau_agg['BUREAU_ACT_BUREAU_IS_DPD_SUM']/bureau_agg['BUREAU_SK_ID_BUREAU_COUNT']
    bureau_agg['BUREAU_ACT_IS_DPD_OVER120_RATIO'] = bureau_agg['BUREAU_ACT_BUREAU_IS_DPD_OVER120_SUM']/bureau_agg['BUREAU_SK_ID_BUREAU_COUNT']

    bureau_agg = bureau_agg.merge(bureau_bal_agg, on='SK_ID_CURR', how='left')
    #bureau_agg = bureau_agg.merge(bureau_days750_agg, on='SK_ID_CURR', how='left')

    return bureau_agg

# Function for POS-CASH Balance dataset feature engineering
def process_pos(pos: pd.DataFrame) -> pd.DataFrame:

    pos = _ensure_copy(pos)
    cond_over_0 = pos['SK_DPD'] > 0
    cond_100 = (pos['SK_DPD'] < 100) & (pos['SK_DPD'] > 0)
    cond_over_100 = (pos['SK_DPD'] >= 100)

    pos['POS_IS_DPD'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    pos['POS_IS_DPD_UNDER_120'] = pos['SK_DPD'].apply(lambda x:1 if (x > 0) & (x <120) else 0 )
    pos['POS_IS_DPD_OVER_120'] = pos['SK_DPD'].apply(lambda x:1 if x >= 120 else 0)

    pos_grp = pos.groupby('SK_ID_CURR')
    pos_agg_dict = {
        'SK_ID_CURR':['count'],
        'MONTHS_BALANCE':['min', 'mean', 'max'],
        'SK_DPD':['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT':['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE':['min', 'max', 'mean', 'sum'],

        'POS_IS_DPD':['mean', 'sum'],
        'POS_IS_DPD_UNDER_120':['mean', 'sum'],
        'POS_IS_DPD_OVER_120':['mean', 'sum']
    }

    pos_agg = pos_grp.agg(pos_agg_dict)

    pos_agg.columns = [('POS_')+('_').join(column).upper() for column in pos_agg.columns.ravel()]

    cond_months = pos['MONTHS_BALANCE'] > -20
    pos_m20_grp = pos[cond_months].groupby('SK_ID_CURR')
    pos_m20_agg_dict = {
        'SK_ID_CURR':['count'],
        'MONTHS_BALANCE':['min', 'mean', 'max'],
        'SK_DPD':['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT':['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE':['min', 'max', 'mean', 'sum'],

        'POS_IS_DPD':['mean', 'sum'],
        'POS_IS_DPD_UNDER_120':['mean', 'sum'],
        'POS_IS_DPD_OVER_120':['mean', 'sum']
    }

    pos_m20_agg = pos_m20_grp.agg(pos_m20_agg_dict)

    pos_m20_agg.columns = [('POS_M20')+('_').join(column).upper() for column in pos_m20_agg.columns.ravel()]
    pos_agg = pos_agg.merge(pos_m20_agg, on='SK_ID_CURR', how='left')
    pos_agg = pos_agg.reset_index()


    return pos_agg

# Function for Installments Payments dataset feature engineering
def process_install(install: pd.DataFrame) -> pd.DataFrame:

    install = _ensure_copy(install)
    install['AMT_DIFF'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
    install['AMT_RATIO'] =  (install['AMT_PAYMENT'] +1)/ (install['AMT_INSTALMENT'] + 1)
    install['SK_DPD'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']

    install['INS_IS_DPD'] = install['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    install['INS_IS_DPD_UNDER_120'] = install['SK_DPD'].apply(lambda x:1 if (x > 0) & (x <120) else 0 )
    install['INS_IS_DPD_OVER_120'] = install['SK_DPD'].apply(lambda x:1 if x >= 120 else 0)

    install_grp = install.groupby('SK_ID_CURR')

    install_agg_dict = {
        'SK_ID_CURR':['count'],
        'NUM_INSTALMENT_VERSION':['nunique'],
        'DAYS_ENTRY_PAYMENT':['mean', 'max', 'sum'],
        'DAYS_INSTALMENT':['mean', 'max', 'sum'],
        'AMT_INSTALMENT':['mean', 'max', 'sum'],
        'AMT_PAYMENT':['mean', 'max','sum'],

        'AMT_DIFF':['mean','min', 'max','sum'],
        'AMT_RATIO':['mean', 'max'],
        'SK_DPD':['mean', 'min', 'max'],
        'INS_IS_DPD':['mean', 'sum'],
        'INS_IS_DPD_UNDER_120':['mean', 'sum'],
        'INS_IS_DPD_OVER_120':['mean', 'sum']
    }

    install_agg = install_grp.agg(install_agg_dict)
    install_agg.columns = ['INS_'+('_').join(column).upper() for column in install_agg.columns.ravel()]

    cond_day = install['DAYS_ENTRY_PAYMENT'] >= -365
    install_d365_grp = install[cond_day].groupby('SK_ID_CURR')
    install_d365_agg_dict = {
        'SK_ID_CURR':['count'],
        'NUM_INSTALMENT_VERSION':['nunique'],
        'DAYS_ENTRY_PAYMENT':['mean', 'max', 'sum'],
        'DAYS_INSTALMENT':['mean', 'max', 'sum'],
        'AMT_INSTALMENT':['mean', 'max', 'sum'],
        'AMT_PAYMENT':['mean', 'max','sum'],

        'AMT_DIFF':['mean','min', 'max','sum'],
        'AMT_RATIO':['mean', 'max'],
        'SK_DPD':['mean', 'min', 'max'],
        'INS_IS_DPD':['mean', 'sum'],
        'INS_IS_DPD_UNDER_120':['mean', 'sum'],
        'INS_IS_DPD_OVER_120':['mean', 'sum']
    }

    install_d365_agg = install_d365_grp.agg(install_d365_agg_dict)
    install_d365_agg.columns = ['INS_D365'+('_').join(column).upper() for column in install_d365_agg.columns.ravel()]

    install_agg = install_agg.merge(install_d365_agg, on='SK_ID_CURR', how='left')
    install_agg = install_agg.reset_index()

    return install_agg

# Function for Credit Card Balance dataset feature engineering
def process_card(card: pd.DataFrame) -> pd.DataFrame:
    card = _ensure_copy(card)

    card['BALANCE_LIMIT_RATIO'] = card['AMT_BALANCE']/card['AMT_CREDIT_LIMIT_ACTUAL']
    card['DRAWING_LIMIT_RATIO'] = card['AMT_DRAWINGS_CURRENT'] / card['AMT_CREDIT_LIMIT_ACTUAL']

    card['CARD_IS_DPD'] = card['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    card['CARD_IS_DPD_UNDER_120'] = card['SK_DPD'].apply(lambda x:1 if (x > 0) & (x <120) else 0 )
    card['CARD_IS_DPD_OVER_120'] = card['SK_DPD'].apply(lambda x:1 if x >= 120 else 0)

    card_grp = card.groupby('SK_ID_CURR')
    card_agg_dict = {
        'SK_ID_CURR':['count'],
        'AMT_BALANCE':['max'],
        'AMT_CREDIT_LIMIT_ACTUAL':['max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max','sum'],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max','sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean'],
        'SK_DPD': ['mean', 'max', 'sum'],

        'BALANCE_LIMIT_RATIO':['min','max'],
        'DRAWING_LIMIT_RATIO':['min', 'max'],
        'CARD_IS_DPD':['mean', 'sum'],
        'CARD_IS_DPD_UNDER_120':['mean', 'sum'],
        'CARD_IS_DPD_OVER_120':['mean', 'sum']
    }
    card_agg = card_grp.agg(card_agg_dict)
    card_agg.columns = ['CARD_'+('_').join(column).upper() for column in card_agg.columns.ravel()]

    card_agg = card_agg.reset_index()

    cond_month = card.MONTHS_BALANCE >= -3
    card_m3_grp = card[cond_month].groupby('SK_ID_CURR')
    card_m3_agg = card_m3_grp.agg(card_agg_dict)
    card_m3_agg.columns = ['CARD_M3'+('_').join(column).upper() for column in card_m3_agg.columns.ravel()]

    card_agg = card_agg.merge(card_m3_agg, on='SK_ID_CURR', how='left')
    card_agg = card_agg.reset_index()

    return card_agg


def behaviorial_features(uci: pd.DataFrame) -> pd.DataFrame:
    """
    Create behavioral features from UCI Credit Card dataset (Behavioral Model).
    
    Engineers 31 features analyzing payment behavior, spending patterns, and
    financial stress indicators from 6 months of credit card history.
    
    Parameters:
    -----------
    uci : pd.DataFrame
        UCI Credit Card data with columns:
        - LIMIT_BAL: Credit limit
        - SEX, EDUCATION, MARRIAGE, AGE: Demographics
        - PAY_0 to PAY_6: Payment status history (0=on-time, 1+=months late)
        - BILL_AMT1 to BILL_AMT6: Monthly bill amounts
        - PAY_AMT1 to PAY_AMT6: Monthly payment amounts
    
    Returns:
    --------
    pd.DataFrame
        Original data + 31 engineered behavioral features
        
    Feature Categories:
    -------------------
    
    1. AGGREGATE FEATURES (5):
       - total_billed_amount: Sum of all bills
       - total_payment_amount: Sum of all payments
       - avg_transaction_amount: Average monthly spending
       - max_billed_amount: Highest single bill
       - max_payment_amount: Largest payment made
    
    2. VOLATILITY & CONSISTENCY (8):
       - spending_volatility: Variability in bills (std)
       - income_consistency: Variability in payments (std)
       - bill_change_1_2 to bill_change_5_6: Month-to-month balance changes
       - rolling_balance_volatility: Volatility of balance changes
    
    3. FINANCIAL STRESS INDICATORS (4):
       - net_flow_balance: Bills - Payments (debt accumulation)
       - debt_stress_index: Bills / Payments ratio
       - repayment_ratio: Payments / Bills (repayment capacity)
       - missed_payment_count: Months with zero payment
    
    4. BEHAVIORAL RATIOS (3):
       - payment_consistency_ratio: Payment volatility normalized by amount
       - spend_to_income_volatility_ratio: Spending vs income variability
       - max_to_mean_bill_ratio: Impulsive spending indicator
    
    5. TREND FEATURES (1):
       - credit_utilization_trend: Linear trend in bill amounts over time
    
    Usage:
    ------
    This function is used for the behavioral model which focuses on
    dynamic payment patterns rather than static credit worthiness.
    
    Note:
    -----
    - Creates a copy to avoid SettingWithCopyWarning
    - Adds 1 to denominators to prevent division by zero
    - Returns DataFrame with both original and engineered features
    
    Example:
    --------
    >>> uci_engineered = behaviorial_features(uci_raw_data)
    >>> print(uci_engineered['debt_stress_index'].describe())
    >>> high_risk = uci_engineered[uci_engineered['missed_payment_count'] > 2]
    """
    # Create a copy to avoid SettingWithCopyWarning
    uci = uci.copy()
    
    # AGGREGATE FEATURES

    # Total billed and payment amounts
    uci["total_billed_amount"] = uci[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                                  "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].sum(axis=1)

    uci["total_payment_amount"] = uci[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                                   "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].sum(axis=1)

    # Average monthly billed amount — proxy for typical spending
    uci["avg_transaction_amount"] = uci[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                                     "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)

    # Maximum monthly amounts (high-value activity)
    uci["max_billed_amount"] = uci[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                                "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].max(axis=1)

    uci["max_payment_amount"] = uci[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                                 "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].max(axis=1)


    # VOLATILITY & CONSISTENCY


    # Spending volatility — variation in bills
    uci["spending_volatility"] = uci[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                                  "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].std(axis=1)

    # Income consistency — variation in payments
    uci["income_consistency"] = uci[["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                                 "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].std(axis=1)

    # Rolling balance volatility — volatility of monthly balance changes
    uci["bill_change_1_2"] = uci["BILL_AMT2"] - uci["BILL_AMT1"]
    uci["bill_change_2_3"] = uci["BILL_AMT3"] - uci["BILL_AMT2"]
    uci["bill_change_3_4"] = uci["BILL_AMT4"] - uci["BILL_AMT3"]
    uci["bill_change_4_5"] = uci["BILL_AMT5"] - uci["BILL_AMT4"]
    uci["bill_change_5_6"] = uci["BILL_AMT6"] - uci["BILL_AMT5"]

    uci["rolling_balance_volatility"] = uci[
    ["bill_change_1_2", "bill_change_2_3", "bill_change_3_4",
     "bill_change_4_5", "bill_change_5_6"]
    ].std(axis=1)


    # FINANCIAL STRESS INDICATORS

    # Net flow balance — difference between spending and repayment
    uci["net_flow_balance"] = uci["total_billed_amount"] - uci["total_payment_amount"]

   # Debt stress index — ratio of bills to payments
    uci["debt_stress_index"] = uci["total_billed_amount"] / (uci["total_payment_amount"] + 1)

   # Repayment ratio — proportion of bills actually paid
    uci["repayment_ratio"] = uci["total_payment_amount"] / (uci["total_billed_amount"] + 1)


   # BEHAVIORAL RATIOS


   # Payment consistency ratio — normalized variability in payments
    uci["payment_consistency_ratio"] = uci["income_consistency"] / (uci["total_payment_amount"] + 1)

   # Spend-to-income volatility ratio — how volatile spending is vs. income
    uci["spend_to_income_volatility_ratio"] = uci["spending_volatility"] / (uci["income_consistency"] + 1)

   # Max-to-mean bill ratio — detect high-value or impulsive spending
    uci["max_to_mean_bill_ratio"] = uci["max_billed_amount"] / (uci["avg_transaction_amount"] + 1)


   # REPAYMENT BEHAVIOR (if PAY_AMT data has zeros)


    # Missed payment count — months with no payment activity
    uci["missed_payment_count"] = (uci[["PAY_AMT1","PAY_AMT2","PAY_AMT3",
                                    "PAY_AMT4","PAY_AMT5","PAY_AMT6"]] == 0).sum(axis=1)


    # TREND FEATURES

    # Credit utilization trend (slope over months)
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]

    # Compute slope (trend) of bill amounts over time for each user
    def compute_slope(row):
        months = np.arange(1, 7).reshape(-1, 1)
        model = LinearRegression().fit(months, row[bill_cols].values)
        return model.coef_[0]

    uci["credit_utilization_trend"] = uci.apply(compute_slope, axis=1)

    # CLEAN-UP
    # Drop intermediate change columns to keep dataset tidy
    uci.drop(columns=["bill_change_1_2", "bill_change_2_3", "bill_change_3_4", 
                      "bill_change_4_5", "bill_change_5_6"], inplace=True)
    
    return uci
