# sme_credit_model/src/data_processing/cleaner.py
# Cleaning functions with column selection

import pandas as pd
import numpy as np
from typing import List, Optional

# Define column lists
APPLICATION_COLS_TO_CLEAN = [
    'company_id', 'company_name', 'application_date', 'balance_sheet_date',
    'revenue', 'total_assets', 'total_liabilities', 'equity',
    'current_ratio', 'quick_ratio', 'debt_to_equity', 'return_on_assets',
    'return_on_equity', 'ebitda', 'ebitda_margin', 'net_profit_margin',
    'interest_coverage', 'debt_service_coverage', 'working_capital',
    'cash_ratio', 'asset_turnover', 'inventory_turnover',
    # Add all your columns here...
]

PORTFOLIO_COLS_TO_CLEAN = [
    'company_id', 'loan_id', 'application_date', 'payout_date',
    'default_date', 'loan_amount', 'interest_rate', 'term_months',
    'days_past_due', 'recovery_amount', 'write_off_date',
    # Add portfolio columns...
]

def clean_application_data(df: pd.DataFrame, 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean application data for specified columns
    
    Args:
        df: Input dataframe
        columns: List of columns to clean (default: APPLICATION_COLS_TO_CLEAN)
    """
    df_clean = df.copy()
    cols_to_clean = columns or APPLICATION_COLS_TO_CLEAN
    
    # Only process columns that exist in the dataframe
    existing_cols = [col for col in cols_to_clean if col in df_clean.columns]
    
    for col in existing_cols:
        df_clean[col] = clean_column(df_clean[col], col)
    
    return df_clean

def clean_portfolio_data(df: pd.DataFrame,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean portfolio data for specified columns
    """
    df_clean = df.copy()
    cols_to_clean = columns or PORTFOLIO_COLS_TO_CLEAN
    
    existing_cols = [col for col in cols_to_clean if col in df_clean.columns]
    
    for col in existing_cols:
        df_clean[col] = clean_column(df_clean[col], col)
    
    return df_clean

def clean_column(series: pd.Series, col_name: str) -> pd.Series:
    """
    Apply specific cleaning based on column name/type
    """
    # Date columns
    if 'date' in col_name.lower():
        return pd.to_datetime(series, errors='coerce')
    
    # ID columns
    elif 'id' in col_name.lower() or col_name == 'company_name':
        return series.astype(str).str.strip().str.upper()
    
    # Financial ratios (handle infinities and outliers)
    elif 'ratio' in col_name.lower() or col_name in ['return_on_assets', 'return_on_equity']:
        series = pd.to_numeric(series, errors='coerce')
        series = series.replace([np.inf, -np.inf], np.nan)
        # Cap extreme values (e.g., ratios > 1000% or < -1000%)
        series = series.clip(lower=-10, upper=10)
        return series
    
    # Monetary amounts
    elif col_name in ['revenue', 'total_assets', 'total_liabilities', 'equity', 
                      'loan_amount', 'recovery_amount', 'ebitda', 'working_capital']:
        series = pd.to_numeric(series, errors='coerce')
        # Remove negative values where not logical
        if col_name in ['total_assets', 'loan_amount']:
            series = series.clip(lower=0)
        return series
    
    # Percentages
    elif 'margin' in col_name.lower() or 'coverage' in col_name.lower():
        series = pd.to_numeric(series, errors='coerce')
        # Convert to decimal if > 1 (assuming it's in percentage form)
        if series.max() > 1:
            series = series / 100
        return series
    
    # Default: convert to numeric if possible
    else:
        return pd.to_numeric(series, errors='ignore')

def standardize_company_ids(df: pd.DataFrame, 
                           id_column: str = 'company_id') -> pd.DataFrame:
    """
    Standardize company IDs for merging
    """
    df_clean = df.copy()
    if id_column in df_clean.columns:
        df_clean[id_column] = (
            df_clean[id_column]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(r'[^A-Z0-9]', '', regex=True)  # Remove special chars
        )
    return df_clean

def calculate_dpd_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate days past due fields
    """
    df_clean = df.copy()
    
    if 'payout_date' in df.columns and 'default_date' in df.columns:
        df_clean['payout_date'] = pd.to_datetime(df_clean['payout_date'])
        df_clean['default_date'] = pd.to_datetime(df_clean['default_date'])
        
        # Calculate days to default
        df_clean['days_to_default'] = (
            df_clean['default_date'] - df_clean['payout_date']
        ).dt.days
        
        # Create DPD categories
        df_clean['dpd_category'] = pd.cut(
            df_clean.get('days_past_due', df_clean['days_to_default']),
            bins=[-np.inf, 0, 30, 60, 90, np.inf],
            labels=['Current', '1-30', '31-60', '61-90', '90+']
        )
    
    return df_clean

# Alternative: Clean multiple column groups at once
def clean_columns_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean columns by type patterns
    """
    df_clean = df.copy()
    
    # Define column groups
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    ratio_cols = [col for col in df.columns if 'ratio' in col.lower()]
    amount_cols = [col for col in df.columns if any(term in col.lower() 
                  for term in ['amount', 'revenue', 'assets', 'liabilities'])]
    
    # Apply cleaning by type
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    for col in id_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()
    
    for col in ratio_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    for col in amount_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean
