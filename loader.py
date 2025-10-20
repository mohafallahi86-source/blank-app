"""
Data Loader Module for SME Credit Acceptance Model
Handles loading of merged application data and portfolio datasets with proper validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and validates data for SME Credit Acceptance Model development
    Uses pre-merged application data from merger.py
    
    Attributes:
        data_dir: Path to data directory
        raw_dir: Path to raw data directory
        processed_dir: Path to processed data directory
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.stats = {}
        
        self._validate_directories()
    
    def _validate_directories(self) -> None:
        """Validate that required directories exist"""
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
        if not self.processed_dir.exists():
            logger.warning(f"Processed data directory not found: {self.processed_dir}")
            self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_merged_applications(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load pre-merged application dataset (output from merger.py)
        
        Args:
            file_path: Optional custom path to merged application file
            
        Returns:
            DataFrame containing merged application data
        """
        if file_path is None:
            file_path = self.processed_dir / "merged_application.xlsx"
        
        logger.info(f"Loading pre-merged applications from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Merged application file not found: {file_path}\n"
                f"Please run merge_application_files() from merger.py first"
            )
        
        if file_path.suffix == '.pkl':
            df = pd.read_pickle(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        logger.info(f"Loaded {len(df):,} application records")
        
        if 'companyID' in df.columns:
            unique_companies = df['companyID'].nunique()
            logger.info(f"Unique companies: {unique_companies:,}")
            logger.info(f"Avg records per company: {len(df)/unique_companies:.2f}")
        
        self.stats['n_applications_loaded'] = len(df)
        
        return df
    
    def load_portfolio(self, portfolio_name: str) -> pd.DataFrame:
        """
        Load portfolio dataset (A or B)
        
        Args:
            portfolio_name: Name of portfolio ('A' or 'B')
            
        Returns:
            DataFrame containing portfolio data
        """
        valid_portfolios = ['A', 'B']
        if portfolio_name not in valid_portfolios:
            raise ValueError(f"Invalid portfolio name: {portfolio_name}. Must be one of {valid_portfolios}")
        
        file_path = self.raw_dir / f"{portfolio_name}_portfolio.xlsx"
        
        logger.info(f"Loading portfolio {portfolio_name} from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Portfolio file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        logger.info(f"Loaded portfolio {portfolio_name}: {len(df):,} loans")
        
        self._validate_portfolio_columns(df, portfolio_name)
        
        self.stats[f'n_portfolio_{portfolio_name}_loaded'] = len(df)
        
        return df
    
    def _validate_portfolio_columns(self, df: pd.DataFrame, portfolio_name: str) -> None:
        """
        Validate that required columns exist in portfolio
        
        Args:
            df: Portfolio DataFrame
            portfolio_name: Name of portfolio
        """
        required_columns = ['payout_date', 'default_date', 'companyID']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Portfolio {portfolio_name} missing expected columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")
        
        if 'payout_date' in df.columns:
            try:
                min_date = pd.to_datetime(df['payout_date']).min()
                max_date = pd.to_datetime(df['payout_date']).max()
                logger.info(f"Portfolio {portfolio_name} date range: {min_date} to {max_date}")
            except:
                logger.warning(f"Could not parse payout_date in portfolio {portfolio_name}")
        
        if 'default_date' in df.columns:
            default_count = df['default_date'].notna().sum()
            default_rate = default_count / len(df) * 100
            logger.info(f"Portfolio {portfolio_name} defaults: {default_count:,} ({default_rate:.2f}%)")
    
    def load_all_portfolios(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available portfolios
        
        Returns:
            Dictionary with portfolio names as keys and DataFrames as values
        """
        portfolios = {}
        
        for portfolio_name in ['A', 'B']:
            try:
                portfolios[portfolio_name] = self.load_portfolio(portfolio_name)
            except FileNotFoundError:
                logger.warning(f"Portfolio {portfolio_name} not found, skipping")
        
        if not portfolios:
            raise FileNotFoundError("No portfolio files found")
        
        return portfolios
    
    def combine_portfolios(self, portfolios: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple portfolios into single DataFrame
        
        Args:
            portfolios: Dictionary of portfolio DataFrames
            
        Returns:
            Combined portfolio DataFrame with source indicator
        """
        logger.info(f"Combining {len(portfolios)} portfolios")
        
        combined_dfs = []
        for name, df in portfolios.items():
            df_copy = df.copy()
            df_copy['portfolio_source'] = name
            combined_dfs.append(df_copy)
        
        combined = pd.concat(combined_dfs, ignore_index=True, sort=False)
        
        logger.info(f"Combined portfolio: {len(combined):,} total loans")
        logger.info(f"Portfolio distribution:\n{combined['portfolio_source'].value_counts()}")
        
        if 'default_date' in combined.columns:
            total_defaults = combined['default_date'].notna().sum()
            total_default_rate = total_defaults / len(combined) * 100
            logger.info(f"Total defaults: {total_defaults:,} ({total_default_rate:.2f}%)")
        
        self.stats['n_combined_portfolio'] = len(combined)
        
        return combined
    
    def load_all_data(self, 
                      merged_app_file: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both pre-merged application and portfolio data
        
        Args:
            merged_app_file: Optional path to merged application file
            
        Returns:
            Tuple of (application_df, portfolio_df)
        """
        logger.info("=" * 80)
        logger.info("Starting data loading process")
        logger.info("=" * 80)
        
        application_df = self.load_merged_applications(merged_app_file)
        
        portfolios = self.load_all_portfolios()
        portfolio_df = self.combine_portfolios(portfolios)
        
        self._log_data_summary(application_df, portfolio_df)
        
        logger.info("=" * 80)
        logger.info("Data loading completed successfully")
        logger.info("=" * 80)
        
        return application_df, portfolio_df
    
    def _log_data_summary(self, application_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> None:
        """Log summary statistics of loaded data"""
        logger.info("\n" + "=" * 80)
        logger.info("DATA LOADING SUMMARY")
        logger.info("=" * 80)
        
        logger.info("\nApplication Data:")
        logger.info(f"  Total rows: {len(application_df):,}")
        logger.info(f"  Total columns: {len(application_df.columns)}")
        if 'companyID' in application_df.columns:
            logger.info(f"  Unique companies: {application_df['companyID'].nunique():,}")
        if 'application_date' in application_df.columns:
            try:
                app_dates = pd.to_datetime(application_df['application_date'])
                logger.info(f"  Date range: {app_dates.min()} to {app_dates.max()}")
            except:
                pass
        
        logger.info("\nPortfolio Data:")
        logger.info(f"  Total loans: {len(portfolio_df):,}")
        if 'companyID' in portfolio_df.columns:
            logger.info(f"  Unique companies: {portfolio_df['companyID'].nunique():,}")
        if 'default_date' in portfolio_df.columns:
            default_count = portfolio_df['default_date'].notna().sum()
            default_rate = default_count / len(portfolio_df) * 100
            logger.info(f"  Total defaults: {default_count:,} ({default_rate:.2f}%)")
        
        logger.info("\nPotential Merge Statistics:")
        if 'companyID' in application_df.columns and 'companyID' in portfolio_df.columns:
            common_companies = set(application_df['companyID']) & set(portfolio_df['companyID'])
            logger.info(f"  Common companies: {len(common_companies):,}")
            expected_merge_rate = len(common_companies) / application_df['companyID'].nunique() * 100
            logger.info(f"  Expected merge rate: {expected_merge_rate:.2f}%")
        
        logger.info("=" * 80)
    
    def validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, any]:
        """
        Perform basic data quality checks
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"\nData quality validation for {dataset_name}")
        logger.info("-" * 80)
        
        validation_results = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_rows': 0,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        missing_summary = df.isnull().sum()
        missing_pct = (missing_summary / len(df) * 100).round(2)
        validation_results['missing_values'] = dict(zip(missing_summary.index, missing_pct))
        
        validation_results['duplicate_rows'] = df.duplicated().sum()
        
        logger.info(f"Total rows: {validation_results['total_rows']:,}")
        logger.info(f"Total columns: {validation_results['total_columns']}")
        logger.info(f"Duplicate rows: {validation_results['duplicate_rows']:,}")
        logger.info(f"Memory usage: {validation_results['memory_usage_mb']:.2f} MB")
        
        high_missing = {col: pct for col, pct in validation_results['missing_values'].items() if pct > 50}
        if high_missing:
            logger.warning(f"\nColumns with >50% missing values: {len(high_missing)}")
            for col, pct in sorted(high_missing.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.warning(f"  {col}: {pct:.2f}%")
        
        return validation_results
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to processed directory
        
        Args:
            df: DataFrame to save
            filename: Name of output file
        """
        output_path = self.processed_dir / filename
        
        logger.info(f"Saving processed data to {output_path}")
        
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        elif output_path.suffix in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=False)
        elif output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.pkl':
            df.to_pickle(output_path)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        logger.info(f"Saved {len(df):,} rows to {output_path}")
    
    def get_stats(self) -> Dict:
        """Return loading statistics"""
        return self.stats


def main():
    """Example usage of DataLoader"""
    loader = DataLoader(data_dir="data")
    
    application_df, portfolio_df = loader.load_all_data()
    
    loader.validate_data_quality(application_df, "Application Data")
    loader.validate_data_quality(portfolio_df, "Portfolio Data")
    
    print("\nLoading Statistics:")
    for key, value in loader.get_stats().items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
