"""
Derived Features Module
Calculates year-over-year changes and aggregates 2 rows into 1 row per application
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DerivedFeatureCalculator:
    """
    Calculates YoY features and aggregates 2 rows (2 years) into 1 row per application
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.stats = {}
    
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            'yoy_features': {
                'absolute_changes': ['revenue', 'profit', 'ebitda', 'total_assets', 'total_debt'],
                'relative_changes': ['revenue', 'profit', 'assets', 'debt', 'equity'],
                'ratio_changes': ['current_ratio', 'debt_to_equity', 'roe', 'roa']
            }
        }
    
    def calculate_yoy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate year-over-year features and aggregate to single row per company
        
        Args:
            df: DataFrame with 2 rows per companyID (sorted newest first)
            
        Returns:
            DataFrame with 1 row per companyID containing base + YoY features
        """
        logger.info("Starting YoY feature calculation")
        logger.info(f"Input: {len(df)} rows, {df['companyID'].nunique()} companies")
        
        # Validate input
        self._validate_input(df)
        
        # Separate into recent (t0) and previous (t1) year
        df_t0, df_t1 = self._split_by_year(df)
        
        # Calculate YoY changes
        yoy_features = self._calculate_changes(df_t0, df_t1)
        
        # Merge with recent year base features
        df_final = self._merge_features(df_t0, yoy_features)
        
        # Log statistics
        self._log_yoy_stats(df_final)
        
        logger.info(f"Output: {len(df_final)} rows (1 per company)")
        
        return df_final
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input data has correct structure"""
        if 'companyID' not in df.columns:
            raise ValueError("companyID column required")
        
        if 'year_indicator' not in df.columns:
            raise ValueError("year_indicator column required (run base_features.py first)")
        
        # Check distribution
        year_dist = df['year_indicator'].value_counts()
        logger.info(f"Year distribution: {dict(year_dist)}")
        
        if 't0' not in year_dist or 't1' not in year_dist:
            raise ValueError("Both t0 and t1 years required")
    
    def _split_by_year(self, df: pd.DataFrame) -> tuple:
        """Split data into recent (t0) and previous (t1) year"""
        logger.info("Splitting data by year")
        
        df_t0 = df[df['year_indicator'] == 't0'].copy()
        df_t1 = df[df['year_indicator'] == 't1'].copy()
        
        logger.info(f"Recent year (t0): {len(df_t0)} rows")
        logger.info(f"Previous year (t1): {len(df_t1)} rows")
        
        # Companies with both years
        companies_t0 = set(df_t0['companyID'])
        companies_t1 = set(df_t1['companyID'])
        companies_both = companies_t0 & companies_t1
        
        pct = len(companies_both) / len(companies_t0) * 100
        logger.info(f"Companies with both years: {len(companies_both)} ({pct:.2f}%)")
        
        self.stats['companies_with_both_years'] = len(companies_both)
        self.stats['companies_t0_only'] = len(companies_t0 - companies_t1)
        
        return df_t0, df_t1
    
    def _calculate_changes(self, df_t0: pd.DataFrame, df_t1: pd.DataFrame) -> pd.DataFrame:
        """Calculate YoY changes"""
        logger.info("Calculating YoY changes")
        
        # Merge t0 and t1 on companyID
        df_merged = df_t0.merge(
            df_t1[['companyID'] + self._get_numeric_columns(df_t1)],
            on='companyID',
            how='left',
            suffixes=('_t0', '_t1')
        )
        
        yoy_df = pd.DataFrame({'companyID': df_merged['companyID']})
        
        # Get columns that exist in both years
        t0_cols = [c for c in df_merged.columns if c.endswith('_t0')]
        base_cols = [c.replace('_t0', '') for c in t0_cols]
        
        for base_col in base_cols:
            col_t0 = f"{base_col}_t0"
            col_t1 = f"{base_col}_t1"
            
            if col_t0 in df_merged.columns and col_t1 in df_merged.columns:
                # Absolute change
                yoy_df[f"{base_col}_change"] = df_merged[col_t0] - df_merged[col_t1]
                
                # Relative change (growth rate)
                with np.errstate(divide='ignore', invalid='ignore'):
                    yoy_df[f"{base_col}_growth_rate"] = (
                        (df_merged[col_t0] - df_merged[col_t1]) / df_merged[col_t1].abs()
                    )
                
                # Trend indicator (-1, 0, 1)
                yoy_df[f"{base_col}_trend"] = np.sign(df_merged[col_t0] - df_merged[col_t1])
        
        logger.info(f"Created {len(yoy_df.columns)-1} YoY features")
        
        return yoy_df
    
    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numeric columns excluding IDs and indicators"""
        exclude = ['companyID', 'year_rank', 'year_indicator', 'application_date', 
                   'financial_statement_date', 'year']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return [c for c in numeric_cols if c not in exclude]
    
    def _merge_features(self, df_t0: pd.DataFrame, yoy_df: pd.DataFrame) -> pd.DataFrame:
        """Merge recent year base features with YoY features"""
        logger.info("Merging base and YoY features")
        
        # Keep only recent year base features + metadata
        keep_cols = ['companyID', 'application_date', 'financial_statement_date', 
                     'industry_code', 'plz_code', 'company_age', 'schufa_score']
        
        # Add all numeric columns from t0
        numeric_cols = self._get_numeric_columns(df_t0)
        keep_cols.extend([c for c in numeric_cols if c not in keep_cols])
        
        # Filter to available columns
        keep_cols = [c for c in keep_cols if c in df_t0.columns]
        
        df_base = df_t0[keep_cols].copy()
        
        # Merge with YoY features
        df_final = df_base.merge(yoy_df, on='companyID', how='left')
        
        logger.info(f"Final dataset: {len(df_final)} rows, {len(df_final.columns)} columns")
        
        return df_final
    
    def _log_yoy_stats(self, df: pd.DataFrame) -> None:
        """Log YoY feature statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("YOY FEATURE CALCULATION STATISTICS")
        logger.info("=" * 80)
        
        # Count feature types
        change_features = [c for c in df.columns if c.endswith('_change')]
        growth_features = [c for c in df.columns if c.endswith('_growth_rate')]
        trend_features = [c for c in df.columns if c.endswith('_trend')]
        
        logger.info(f"\nFeature counts:")
        logger.info(f"  Absolute changes: {len(change_features)}")
        logger.info(f"  Growth rates: {len(growth_features)}")
        logger.info(f"  Trend indicators: {len(trend_features)}")
        logger.info(f"  Total YoY features: {len(change_features) + len(growth_features) + len(trend_features)}")
        
        # Example growth rates
        if growth_features:
            logger.info(f"\nExample growth rate statistics:")
            for feat in growth_features[:5]:
                mean = df[feat].mean()
                median = df[feat].median()
                logger.info(f"  {feat}: mean={mean:.2f}, median={median:.2f}")
        
        # Missing value analysis for YoY features
        logger.info(f"\nMissing values in YoY features:")
        for feat in change_features[:5]:
            missing_pct = df[feat].isnull().sum() / len(df) * 100
            logger.info(f"  {feat}: {missing_pct:.2f}%")
        
        logger.info("=" * 80)
    
    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk indicators
        
        Args:
            df: DataFrame with base and YoY features
            
        Returns:
            DataFrame with additional composite features
        """
        logger.info("Creating composite features")
        
        df_comp = df.copy()
        
        # Altman Z-Score (simplified for private companies)
        if all(col in df.columns for col in ['working_capital', 'total_assets', 'roe', 'equity_ratio']):
            df_comp['altman_z_score'] = (
                1.2 * (df['working_capital'] / df['total_assets']) +
                1.4 * (df['roe'] / 100) +
                3.3 * (df['roa'] / 100) if 'roa' in df.columns else 0 +
                0.6 * (df['equity_ratio'] / 100)
            )
        
        # Liquidity index (composite of liquidity ratios)
        liquidity_cols = ['current_ratio', 'quick_ratio', 'cash_ratio']
        available_liq = [c for c in liquidity_cols if c in df.columns]
        if available_liq:
            df_comp['liquidity_index'] = df[available_liq].mean(axis=1)
        
        # Leverage index
        leverage_cols = ['debt_to_equity', 'debt_to_assets', 'equity_ratio']
        available_lev = [c for c in leverage_cols if c in df.columns]
        if available_lev:
            df_comp['leverage_index'] = df[available_lev].mean(axis=1)
        
        # Profitability index
        profit_cols = ['roe', 'roa', 'ros', 'net_profit_margin']
        available_prof = [c for c in profit_cols if c in df.columns]
        if available_prof:
            df_comp['profitability_index'] = df[available_prof].mean(axis=1)
        
        # Growth momentum (average of growth rates)
        growth_cols = [c for c in df.columns if c.endswith('_growth_rate')]
        if growth_cols:
            df_comp['growth_momentum'] = df[growth_cols].mean(axis=1)
        
        # Financial distress score (negative trends)
        trend_cols = [c for c in df.columns if c.endswith('_trend')]
        if trend_cols:
            df_comp['negative_trends_count'] = (df[trend_cols] < 0).sum(axis=1)
            df_comp['positive_trends_count'] = (df[trend_cols] > 0).sum(axis=1)
        
        logger.info(f"Created {len(df_comp.columns) - len(df.columns)} composite features")
        
        return df_comp


def main():
    """Example usage"""
    from pathlib import Path
    
    # Initialize calculator
    config_path = Path("config/feature_config.yaml")
    if config_path.exists():
        calculator = DerivedFeatureCalculator(str(config_path))
    else:
        logger.warning("Config file not found, using defaults")
        calculator = DerivedFeatureCalculator()
    
    # Load base features
    input_path = Path("data/features/base_features.parquet")
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run base_features.py first")
        return
    
    df = pd.read_parquet(input_path)
    
    # Calculate YoY features
    df_yoy = calculator.calculate_yoy_features(df)
    
    # Create composite features
    df_final = calculator.create_composite_features(df_yoy)
    
    # Save
    output_path = Path("data/features/derived_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False)
    logger.info(f"Saved derived features to {output_path}")
    
    logger.info(f"\nFinal dataset shape: {df_final.shape}")
    logger.info(f"Features created: {len(df_final.columns)}")


if __name__ == "__main__":
    main()
