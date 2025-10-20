"""
Base Features Module
Extracts and validates base financial features from vendor-processed application data
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseFeatureExtractor:
    """
    Extracts base financial features from vendor-processed application data
    Each application has 2 rows (2 years of financial data)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.base_feature_list = self._get_base_feature_list()
        self.feature_metadata = {}
    
    def _get_default_config(self) -> Dict:
        """Default configuration if no config file provided"""
        return {
            'base_features': {
                'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio', 'working_capital'],
                'leverage': ['debt_to_equity', 'equity_ratio', 'debt_to_assets'],
                'profitability': ['roe', 'roa', 'ros', 'net_profit_margin'],
                'efficiency': ['asset_turnover', 'inventory_turnover'],
                'growth': ['revenue_growth', 'asset_growth'],
                'other': ['company_age', 'industry_code', 'plz_code', 'schufa_score']
            }
        }
    
    def _get_base_feature_list(self) -> List[str]:
        """Flatten base features from config"""
        features = []
        for category, feature_list in self.config['base_features'].items():
            features.extend(feature_list)
        return features
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract base features from application dataframe
        
        Args:
            df: Application dataframe with 2 rows per companyID
            
        Returns:
            DataFrame with extracted and validated features
        """
        logger.info("Starting base feature extraction")
        logger.info(f"Input data: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate input data structure
        self._validate_input_data(df)
        
        # Sort by companyID and year (newest first)
        df_sorted = self._sort_by_company_and_year(df)
        
        # Extract features for each year
        df_with_year_suffix = self._add_year_suffixes(df_sorted)
        
        # Log extraction statistics
        self._log_extraction_stats(df_with_year_suffix)
        
        logger.info("Base feature extraction completed")
        
        return df_with_year_suffix
    
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data structure"""
        logger.info("Validating input data structure")
        
        if 'companyID' not in df.columns:
            raise ValueError("companyID column not found in dataframe")
        
        # Check for required date columns
        date_columns = ['application_date', 'financial_statement_date', 'year']
        available_date_cols = [col for col in date_columns if col in df.columns]
        
        if not available_date_cols:
            logger.warning("No date columns found. Cannot verify temporal ordering.")
        
        # Check number of rows per company
        rows_per_company = df.groupby('companyID').size()
        logger.info(f"Rows per company distribution:\n{rows_per_company.value_counts().head()}")
        
        companies_with_2_rows = (rows_per_company == 2).sum()
        total_companies = df['companyID'].nunique()
        pct = companies_with_2_rows / total_companies * 100
        
        logger.info(f"Companies with exactly 2 rows: {companies_with_2_rows} ({pct:.2f}%)")
        
        if pct < 80:
            logger.warning(f"Only {pct:.2f}% of companies have 2 rows of financial data")
    
    def _sort_by_company_and_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data by companyID and financial year (newest first)"""
        logger.info("Sorting data by company and year")
        
        if 'financial_statement_date' in df.columns:
            df['financial_statement_date'] = pd.to_datetime(df['financial_statement_date'])
            df_sorted = df.sort_values(
                ['companyID', 'financial_statement_date'], 
                ascending=[True, False]
            )
        elif 'year' in df.columns:
            df_sorted = df.sort_values(
                ['companyID', 'year'], 
                ascending=[True, False]
            )
        else:
            logger.warning("No year/date column found. Using original order.")
            df_sorted = df.sort_values('companyID')
        
        return df_sorted
    
    def _add_year_suffixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add year suffixes to features (_t0 for most recent, _t1 for previous year)
        Keep both rows for now (will aggregate later)
        
        Args:
            df: Sorted dataframe
            
        Returns:
            DataFrame with year suffixes
        """
        logger.info("Adding year suffixes to features")
        
        # Add row number within each company group
        df['year_rank'] = df.groupby('companyID').cumcount()
        
        # Create year indicator (0 = most recent, 1 = previous year)
        df['year_indicator'] = df['year_rank'].map({0: 't0', 1: 't1'})
        
        # Identify numeric financial columns (exclude ID, date, categorical columns)
        exclude_cols = ['companyID', 'application_date', 'financial_statement_date', 
                       'year', 'year_rank', 'year_indicator', 'industry_code', 'plz_code']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Identified {len(feature_cols)} numeric feature columns")
        
        # Store metadata about available features
        self.feature_metadata['available_features'] = feature_cols
        self.feature_metadata['missing_features'] = [
            f for f in self.base_feature_list if f not in feature_cols
        ]
        
        if self.feature_metadata['missing_features']:
            logger.warning(
                f"Missing expected features: {self.feature_metadata['missing_features'][:10]}"
            )
        
        return df
    
    def _log_extraction_stats(self, df: pd.DataFrame) -> None:
        """Log statistics about extracted features"""
        logger.info("\n" + "=" * 80)
        logger.info("BASE FEATURE EXTRACTION STATISTICS")
        logger.info("=" * 80)
        
        logger.info(f"\nTotal rows: {len(df):,}")
        logger.info(f"Unique companies: {df['companyID'].nunique():,}")
        
        if 'year_indicator' in df.columns:
            logger.info(f"\nYear distribution:\n{df['year_indicator'].value_counts()}")
        
        # Check for available feature categories
        for category, features in self.config['base_features'].items():
            available = [f for f in features if f in df.columns]
            pct = len(available) / len(features) * 100 if features else 0
            logger.info(f"\n{category.upper()}: {len(available)}/{len(features)} features ({pct:.1f}%)")
            
            if len(available) < len(features):
                missing = [f for f in features if f not in df.columns]
                logger.warning(f"  Missing: {missing}")
        
        # Missing value statistics for key features
        logger.info("\nMissing value percentages for key features:")
        key_features = ['current_ratio', 'debt_to_equity', 'roe', 'roa', 
                       'revenue_growth', 'schufa_score']
        
        for feature in key_features:
            if feature in df.columns:
                missing_pct = df[feature].isnull().sum() / len(df) * 100
                logger.info(f"  {feature}: {missing_pct:.2f}%")
        
        logger.info("=" * 80)
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for all base features
        
        Args:
            df: DataFrame with features
            
        Returns:
            Summary DataFrame with statistics
        """
        logger.info("Generating feature summary statistics")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['companyID', 'year_rank']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        summary_stats = []
        
        for col in feature_cols:
            stats = {
                'feature': col,
                'count': df[col].count(),
                'missing': df[col].isnull().sum(),
                'missing_pct': df[col].isnull().sum() / len(df) * 100,
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'p25': df[col].quantile(0.25),
                'median': df[col].median(),
                'p75': df[col].quantile(0.75),
                'max': df[col].max(),
                'nunique': df[col].nunique()
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        return summary_df
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate feature quality
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating feature quality")
        
        validation_results = {
            'total_features': 0,
            'high_missing': [],
            'zero_variance': [],
            'extreme_outliers': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['companyID', 'year_rank']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        validation_results['total_features'] = len(feature_cols)
        
        for col in feature_cols:
            # Check missing values
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 50:
                validation_results['high_missing'].append((col, missing_pct))
            
            # Check variance
            if df[col].std() == 0:
                validation_results['zero_variance'].append(col)
            
            # Check extreme outliers (beyond 99.9th percentile by large margin)
            if not df[col].isnull().all():
                p999 = df[col].quantile(0.999)
                max_val = df[col].max()
                if max_val > p999 * 10:  # Max is 10x the 99.9th percentile
                    validation_results['extreme_outliers'].append((col, max_val, p999))
        
        # Log validation results
        logger.info(f"\nValidation Results:")
        logger.info(f"  Total features validated: {validation_results['total_features']}")
        logger.info(f"  Features with >50% missing: {len(validation_results['high_missing'])}")
        logger.info(f"  Zero variance features: {len(validation_results['zero_variance'])}")
        logger.info(f"  Features with extreme outliers: {len(validation_results['extreme_outliers'])}")
        
        return validation_results


def main():
    """Example usage"""
    from pathlib import Path
    
    # Initialize extractor
    config_path = Path("config/feature_config.yaml")
    if config_path.exists():
        extractor = BaseFeatureExtractor(str(config_path))
    else:
        logger.warning("Config file not found, using defaults")
        extractor = BaseFeatureExtractor()
    
    # Load cleaned application data
    input_path = Path("data/processed/cleaned_applications.parquet")
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    df = pd.read_parquet(input_path)
    
    # Extract base features
    df_features = extractor.extract_features(df)
    
    # Generate summary
    summary = extractor.get_feature_summary(df_features)
    print("\nFeature Summary:")
    print(summary.head(20))
    
    # Validate features
    validation = extractor.validate_features(df_features)
    
    # Save extracted features
    output_path = Path("data/features/base_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    logger.info(f"Saved base features to {output_path}")


if __name__ == "__main__":
    main()
