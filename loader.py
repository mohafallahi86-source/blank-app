# sme_credit_model/src/data_processing/loader.py
# Hybrid approach - uses functions from merger.py, handles pre-processed data

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .merger import merge_application_files  # Import but won't use (already done)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SMEDataProcessor:
    """Main processor using pre-merged application data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.stats = {}
        self.processed_data = None
        
    def process_pipeline(self,
                        merged_app_file: Path,
                        portfolio1_file: Path, 
                        portfolio2_file: Path) -> pd.DataFrame:
        """
        Main pipeline using pre-merged data
        
        Args:
            merged_app_file: Path to pre-merged 34k applications (data/processed/...)
            portfolio1_file: Path to first portfolio (data/raw/portfolio_dataset/...)
            portfolio2_file: Path to second portfolio (data/raw/portfolio_dataset/...)
        """
        
        logger.info("Starting pipeline with pre-merged data")
        
        # Step 1: Load pre-merged applications
        applications = self._load_merged_applications(merged_app_file)
        
        # Step 2: Clean and transform applications
        applications_clean = self._clean_applications(applications)
        
        # Step 3: Load portfolio datasets
        portfolio1 = self._load_portfolio(portfolio1_file, "portfolio_1")
        portfolio2 = self._load_portfolio(portfolio2_file, "portfolio_2")
        
        # Step 4: Clean and transform portfolios
        portfolio1_clean = self._clean_portfolio(portfolio1, "portfolio_1")
        portfolio2_clean = self._clean_portfolio(portfolio2, "portfolio_2")
        
        # Step 5: Combine portfolios
        combined_portfolio = self._combine_portfolios(portfolio1_clean, portfolio2_clean)
        
        # Step 6: Merge applications with portfolio
        final_data = self._merge_app_portfolio(applications_clean, combined_portfolio)
        
        self.processed_data = final_data
        return final_data
    
    def _load_merged_applications(self, file_path: Path) -> pd.DataFrame:
        """Load the pre-merged application dataset"""
        logger.info(f"Loading pre-merged applications from {file_path}")
        
        if file_path.suffix == '.pkl':
            df = pd.read_pickle(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        logger.info(f"Loaded {len(df)} applications")
        self.stats['n_applications_loaded'] = len(df)
        return df
    
    def _clean_applications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform application data"""
        logger.info("Cleaning application data")
        
        # TODO: Add your cleaning logic here
        # Examples:
        # - Standardize column names
        # - Parse dates
        # - Handle missing values
        # - Fix data types
        
        df_clean = df.copy()
        
        # Track cleaning stats
        self.stats['n_applications_after_clean'] = len(df_clean)
        return df_clean
    
    def _load_portfolio(self, file_path: Path, name: str) -> pd.DataFrame:
        """Load a portfolio dataset"""
        logger.info(f"Loading {name} from {file_path}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        logger.info(f"Loaded {len(df)} records from {name}")
        self.stats[f'n_{name}_loaded'] = len(df)
        return df
    
    def _clean_portfolio(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Clean and transform portfolio data"""
        logger.info(f"Cleaning {name}")
        
        # TODO: Add portfolio-specific cleaning
        # Examples:
        # - Parse date columns
        # - Calculate DPD
        # - Standardize company IDs
        
        df_clean = df.copy()
        
        self.stats[f'n_{name}_after_clean'] = len(df_clean)
        return df_clean
    
    def _combine_portfolios(self, portfolio1: pd.DataFrame, 
                           portfolio2: pd.DataFrame) -> pd.DataFrame:
        """Combine two portfolio datasets"""
        logger.info("Combining portfolio datasets")
        
        # TODO: Add combination logic
        # Options: concat, merge, or custom logic
        
        combined = pd.concat([portfolio1, portfolio2], ignore_index=True)
        
        logger.info(f"Combined portfolio has {len(combined)} records")
        self.stats['n_combined_portfolio'] = len(combined)
        return combined
    
    def _merge_app_portfolio(self, applications: pd.DataFrame,
                            portfolio: pd.DataFrame) -> pd.DataFrame:
        """Merge applications with portfolio - expecting ~4,114 matches"""
        logger.info("Merging applications with portfolio")
        
        # TODO: Add merge logic
        # Likely merge on company_id and date conditions
        
        merged = applications  # Placeholder
        
        logger.info(f"Final merged dataset: {len(merged)} records")
        self.stats['n_final_merged'] = len(merged)
        self.stats['merge_rate'] = len(merged) / len(applications) * 100
        
        return merged
    
    def get_stats(self) -> Dict:
        """Return processing statistics"""
        return self.stats

# Usage example
if __name__ == "__main__":
    processor = SMEDataProcessor()
    
    result = processor.process_pipeline(
        merged_app_file=Path("data/processed/merged_applications.pkl"),
        portfolio1_file=Path("data/raw/portfolio_dataset/portfolio1.csv"),
        portfolio2_file=Path("data/raw/portfolio_dataset/portfolio2.csv")
    )
    
    print(f"Processing complete: {len(result)} records")
    print(f"Stats: {processor.get_stats()}")
