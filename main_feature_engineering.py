"""
Main Feature Engineering Pipeline
Orchestrates the complete feature engineering process
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import yaml
import json

from base_features import BaseFeatureExtractor
from derived_features import DerivedFeatureCalculator
from time_features import TimeFeatureCreator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline:
    1. Load cleaned data
    2. Extract base features
    3. Calculate YoY features (2 rows -> 1 row)
    4. Create time features
    5. Create composite features
    6. Save engineered features
    """
    
    def __init__(self, config_path: str = "config/feature_config.yaml"):
        self.config_path = Path(config_path)
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}")
            self.config = {}
        
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'duration_seconds': None,
            'steps_completed': [],
            'data_shapes': {}
        }
    
    def run_pipeline(self,
                    input_applications: str = "data/processed/cleaned_applications.parquet",
                    output_features: str = "data/features/engineered_features.parquet") -> pd.DataFrame:
        """
        Run complete feature engineering pipeline
        
        Args:
            input_applications: Path to cleaned applications (2 rows per company)
            output_features: Path to save engineered features (1 row per company)
            
        Returns:
            DataFrame with engineered features
        """
        self.pipeline_stats['start_time'] = datetime.now()
        
        logger.info("=" * 100)
        logger.info("FEATURE ENGINEERING PIPELINE START")
        logger.info("=" * 100)
        logger.info(f"Start time: {self.pipeline_stats['start_time']}")
        logger.info(f"Input: {input_applications}")
        logger.info(f"Output: {output_features}")
        
        try:
            # Step 1: Load data
            df = self._load_data(input_applications)
            
            # Step 2: Extract base features
            df = self._extract_base_features(df)
            
            # Step 3: Calculate YoY features and aggregate to 1 row per company
            df = self._calculate_yoy_features(df)
            
            # Step 4: Create time features
            df = self._create_time_features(df)
            
            # Step 5: Create composite features
            df = self._create_composite_features(df)
            
            # Step 6: Validate and save
            self._validate_features(df)
            self._save_features(df, output_features)
            
            # Generate metadata
            self._generate_metadata(df, output_features)
            
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['duration_seconds'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            ).total_seconds()
            
            self._log_pipeline_summary()
            
            return df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def _load_data(self, input_path: str) -> pd.DataFrame:
        """Load cleaned application data"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Load Data")
        logger.info("=" * 80)
        
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Loading from: {input_path}")
        
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"Unique companies: {df['companyID'].nunique():,}")
        
        self.pipeline_stats['data_shapes']['input'] = df.shape
        self.pipeline_stats['steps_completed'].append('load_data')
        
        return df
    
    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract base financial features"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Extract Base Features")
        logger.info("=" * 80)
        
        extractor = BaseFeatureExtractor(str(self.config_path) if self.config_path.exists() else None)
        df = extractor.extract_features(df)
        
        self.pipeline_stats['data_shapes']['base_features'] = df.shape
        self.pipeline_stats['steps_completed'].append('extract_base_features')
        
        return df
    
    def _calculate_yoy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate YoY features and aggregate to 1 row per company"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Calculate YoY Features")
        logger.info("=" * 80)
        
        calculator = DerivedFeatureCalculator(str(self.config_path) if self.config_path.exists() else None)
        df = calculator.calculate_yoy_features(df)
        
        logger.info(f"Aggregated to 1 row per company: {len(df):,} rows")
        
        self.pipeline_stats['data_shapes']['yoy_features'] = df.shape
        self.pipeline_stats['steps_completed'].append('calculate_yoy_features')
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Create Time Features")
        logger.info("=" * 80)
        
        creator = TimeFeatureCreator(str(self.config_path) if self.config_path.exists() else None)
        df = creator.create_time_features(df)
        
        self.pipeline_stats['data_shapes']['time_features'] = df.shape
        self.pipeline_stats['steps_completed'].append('create_time_features')
        
        return df
    
    def _create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk indicators"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Create Composite Features")
        logger.info("=" * 80)
        
        calculator = DerivedFeatureCalculator(str(self.config_path) if self.config_path.exists() else None)
        df = calculator.create_composite_features(df)
        
        self.pipeline_stats['data_shapes']['composite_features'] = df.shape
        self.pipeline_stats['steps_completed'].append('create_composite_features')
        
        return df
    
    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate engineered features"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Validate Features")
        logger.info("=" * 80)
        
        # Check for required columns
        required_cols = ['companyID']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for duplicates
        duplicate_count = df['companyID'].duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate companyIDs")
        
        # Check missing values
        missing_summary = df.isnull().sum()
        high_missing = missing_summary[missing_summary / len(df) > 0.5]
        
        if len(high_missing) > 0:
            logger.warning(f"Features with >50% missing values: {len(high_missing)}")
            for col, count in high_missing.items():
                pct = count / len(df) * 100
                logger.warning(f"  {col}: {pct:.1f}%")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            logger.warning(f"Features with infinite values: {len(inf_counts)}")
            for col, count in list(inf_counts.items())[:5]:
                logger.warning(f"  {col}: {count} infinite values")
        
        logger.info("Validation completed")
        self.pipeline_stats['steps_completed'].append('validate_features')
    
    def _save_features(self, df: pd.DataFrame, output_path: str) -> None:
        """Save engineered features"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: Save Features")
        logger.info("=" * 80)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to: {output_path}")
        
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        self.pipeline_stats['output_path'] = str(output_path)
        self.pipeline_stats['output_size_mb'] = file_size_mb
        self.pipeline_stats['steps_completed'].append('save_features')
    
    def _generate_metadata(self, df: pd.DataFrame, output_path: str) -> None:
        """Generate feature metadata and documentation"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: Generate Metadata")
        logger.info("=" * 80)
        
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'n_rows': len(df),
            'n_features': len(df.columns) - 1,  # Exclude companyID
            'features': {},
            'pipeline_stats': self.pipeline_stats
        }
        
        # Feature statistics
        for col in df.columns:
            if col == 'companyID':
                continue
            
            feat_meta = {
                'dtype': str(df[col].dtype),
                'missing_pct': float(df[col].isnull().sum() / len(df) * 100),
                'nunique': int(df[col].nunique())
            }
            
            if df[col].dtype in [np.float64, np.int64]:
                feat_meta.update({
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None
                })
            
            metadata['features'][col] = feat_meta
        
        # Save metadata
        metadata_path = Path(output_path).parent / "feature_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to: {metadata_path}")
        self.pipeline_stats['steps_completed'].append('generate_metadata')
    
    def _log_pipeline_summary(self) -> None:
        """Log pipeline summary"""
        logger.info("\n" + "=" * 100)
        logger.info("FEATURE ENGINEERING PIPELINE COMPLETED")
        logger.info("=" * 100)
        
        logger.info(f"\nExecution Summary:")
        logger.info(f"  Start time: {self.pipeline_stats['start_time']}")
        logger.info(f"  End time: {self.pipeline_stats['end_time']}")
        logger.info(f"  Duration: {self.pipeline_stats['duration_seconds']:.2f} seconds")
        
        logger.info(f"\nData Transformations:")
        for step, shape in self.pipeline_stats['data_shapes'].items():
            logger.info(f"  {step}: {shape[0]:,} rows × {shape[1]} columns")
        
        logger.info(f"\nSteps Completed: {len(self.pipeline_stats['steps_completed'])}")
        for i, step in enumerate(self.pipeline_stats['steps_completed'], 1):
            logger.info(f"  {i}. {step}")
        
        logger.info("\n" + "=" * 100)


def main():
    """Run feature engineering pipeline"""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(config_path="config/feature_config.yaml")
    
    # Run pipeline
    df_features = pipeline.run_pipeline(
        input_applications="data/processed/cleaned_applications.parquet",
        output_features="data/features/engineered_features.parquet"
    )
    
    logger.info(f"\nPipeline completed successfully!")
    logger.info(f"Engineered features: {df_features.shape}")
    logger.info(f"Output saved to: data/features/engineered_features.parquet")


if __name__ == "__main__":
    main()
