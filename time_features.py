"""
Time Features Module
Creates time-based features from application and financial statement dates
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeFeatureCreator:
    """
    Creates time-based features including lags, seasonality, and consistency metrics
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        self.stats = {}
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all time-based features
        
        Args:
            df: DataFrame with application_date and financial_statement_date
            
        Returns:
            DataFrame with additional time features
        """
        logger.info("Creating time-based features")
        logger.info(f"Input: {len(df)} rows")
        
        df_time = df.copy()
        
        # Parse dates
        df_time = self._parse_dates(df_time)
        
        # Create lag features
        df_time = self._create_lag_features(df_time)
        
        # Create seasonality features
        df_time = self._create_seasonality_features(df_time)
        
        # Create data age features
        df_time = self._create_age_features(df_time)
        
        # Log statistics
        self._log_time_stats(df_time)
        
        return df_time
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date columns"""
        logger.info("Parsing date columns")
        
        date_cols = ['application_date', 'financial_statement_date', 'payout_date']
        
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.info(f"Parsed {col}")
                except Exception as e:
                    logger.warning(f"Could not parse {col}: {e}")
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features between key dates"""
        logger.info("Creating lag features")
        
        # Application to financial statement lag
        if 'application_date' in df.columns and 'financial_statement_date' in df.columns:
            df['app_to_financial_lag_days'] = (
                df['application_date'] - df['financial_statement_date']
            ).dt.days
            
            df['app_to_financial_lag_months'] = (
                df['app_to_financial_lag_days'] / 30.44
            ).round(1)
            
            # Categorize lag
            df['app_to_financial_lag_category'] = pd.cut(
                df['app_to_financial_lag_months'],
                bins=[-np.inf, 3, 6, 9, 12, np.inf],
                labels=['0-3m', '3-6m', '6-9m', '9-12m', '12m+']
            )
            
            # Log lag statistics
            logger.info(f"Application to financial lag - Mean: {df['app_to_financial_lag_months'].mean():.1f} months")
            logger.info(f"Lag distribution:\n{df['app_to_financial_lag_category'].value_counts()}")
        
        return df
    
    def _create_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonality features"""
        logger.info("Creating seasonality features")
        
        # Financial statement seasonality
        if 'financial_statement_date' in df.columns:
            df['financial_statement_month'] = df['financial_statement_date'].dt.month
            df['financial_statement_quarter'] = df['financial_statement_date'].dt.quarter
            
            # Fiscal year end flag (Q4 often = Dec 31)
            df['is_fiscal_year_end'] = (df['financial_statement_month'] == 12).astype(int)
        
        # Application seasonality
        if 'application_date' in df.columns:
            df['application_month'] = df['application_date'].dt.month
            df['application_quarter'] = df['application_date'].dt.quarter
            df['application_year'] = df['application_date'].dt.year
            
            # Business cycle indicators
            df['application_is_q1'] = (df['application_quarter'] == 1).astype(int)
            df['application_is_q4'] = (df['application_quarter'] == 4).astype(int)
        
        return df
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to data age"""
        logger.info("Creating data age features")
        
        # Age of financial data at application
        if 'application_date' in df.columns and 'financial_statement_date' in df.columns:
            df['financial_data_age_days'] = (
                df['application_date'] - df['financial_statement_date']
            ).dt.days
            
            df['financial_data_age_months'] = (
                df['financial_data_age_days'] / 30.44
            ).round(1)
            
            # Flag stale data (>18 months old)
            df['financial_data_is_stale'] = (
                df['financial_data_age_months'] > 18
            ).astype(int)
        
        # Days between application and payout (if available)
        if 'application_date' in df.columns and 'payout_date' in df.columns:
            df['app_to_payout_days'] = (
                df['payout_date'] - df['application_date']
            ).dt.days
            
            # Flag unusually fast/slow processing
            df['fast_processing'] = (df['app_to_payout_days'] < 7).astype(int)
            df['slow_processing'] = (df['app_to_payout_days'] > 60).astype(int)
        
        return df
    
    def _log_time_stats(self, df: pd.DataFrame) -> None:
        """Log time feature statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("TIME FEATURE STATISTICS")
        logger.info("=" * 80)
        
        time_features = [c for c in df.columns if any(
            x in c.lower() for x in ['lag', 'age', 'month', 'quarter', 'seasonality']
        )]
        
        logger.info(f"\nCreated {len(time_features)} time-based features")
        
        # Lag statistics
        if 'app_to_financial_lag_months' in df.columns:
            lag_stats = df['app_to_financial_lag_months'].describe()
            logger.info(f"\nApplication to Financial Lag (months):")
            logger.info(f"  Mean: {lag_stats['mean']:.2f}")
            logger.info(f"  Median: {lag_stats['50%']:.2f}")
            logger.info(f"  Min: {lag_stats['min']:.2f}")
            logger.info(f"  Max: {lag_stats['max']:.2f}")
            
            if 'app_to_financial_lag_category' in df.columns:
                logger.info(f"\nLag Category Distribution:")
                for cat, count in df['app_to_financial_lag_category'].value_counts().items():
                    pct = count / len(df) * 100
                    logger.info(f"  {cat}: {count:,} ({pct:.1f}%)")
        
        # Data age statistics
        if 'financial_data_age_months' in df.columns:
            age_stats = df['financial_data_age_months'].describe()
            logger.info(f"\nFinancial Data Age (months):")
            logger.info(f"  Mean: {age_stats['mean']:.2f}")
            logger.info(f"  Median: {age_stats['50%']:.2f}")
            
            if 'financial_data_is_stale' in df.columns:
                stale_count = df['financial_data_is_stale'].sum()
                stale_pct = stale_count / len(df) * 100
                logger.info(f"  Stale data (>18 months): {stale_count:,} ({stale_pct:.1f}%)")
        
        # Seasonality distribution
        if 'application_quarter' in df.columns:
            logger.info(f"\nApplication Quarter Distribution:")
            for q, count in df['application_quarter'].value_counts().sort_index().items():
                pct = count / len(df) * 100
                logger.info(f"  Q{q}: {count:,} ({pct:.1f}%)")
        
        logger.info("=" * 80)


def main():
    """Example usage"""
    from pathlib import Path
    
    # Initialize creator
    config_path = Path("config/feature_config.yaml")
    if config_path.exists():
        creator = TimeFeatureCreator(str(config_path))
    else:
        creator = TimeFeatureCreator()
    
    # Load derived features
    input_path = Path("data/features/derived_features.parquet")
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run derived_features.py first")
        return
    
    df = pd.read_parquet(input_path)
    
    # Create time features
    df_time = creator.create_time_features(df)
    
    # Save
    output_path = Path("data/features/time_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_time.to_parquet(output_path, index=False)
    logger.info(f"\nSaved time features to {output_path}")
    
    logger.info(f"\nFinal dataset shape: {df_time.shape}")


if __name__ == "__main__":
    main()
