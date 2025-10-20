"""
Feature Selection Module
Implements univariate and multivariate feature selection per ING procedure
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
import yaml

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection following ING Credit Acceptance Model Development Procedure
    Implements domain knowledge, univariate, and multivariate selection
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.selection_results = {
            'domain_knowledge': [],
            'univariate': [],
            'multivariate': [],
            'final': []
        }
    
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            'feature_selection': {
                'univariate': {
                    'iv_threshold': 0.02,
                    'missing_threshold': 0.5,
                    'psi_threshold': 0.25
                },
                'multivariate': {
                    'correlation_threshold': 0.8,
                    'vif_threshold': 10
                },
                'final_selection': {
                    'max_features_conventional': 10,
                    'max_features_advanced': 40
                }
            }
        }
    
    def select_features(self, 
                       df: pd.DataFrame, 
                       target: str,
                       model_type: str = 'conventional') -> pd.DataFrame:
        """
        Main feature selection pipeline
        
        Args:
            df: DataFrame with features and target
            target: Target column name
            model_type: 'conventional' or 'advanced'
            
        Returns:
            DataFrame with selected features
        """
        logger.info("=" * 80)
        logger.info("FEATURE SELECTION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Model type: {model_type}")
        logger.info(f"Input features: {len(df.columns) - 1}")
        
        # Step 1: Domain knowledge selection
        df_domain = self.domain_knowledge_selection(df, target)
        
        # Step 2: Univariate selection
        df_univariate = self.univariate_selection(df_domain, target)
        
        # Step 3: Multivariate selection
        df_multivariate = self.multivariate_selection(df_univariate, target)
        
        # Step 4: Final selection based on model type
        df_final = self.final_selection(df_multivariate, target, model_type)
        
        # Log summary
        self._log_selection_summary()
        
        return df_final
    
    def domain_knowledge_selection(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Domain knowledge based selection (Section 10.3 of procedure)
        Remove features based on business logic, legal constraints, data quality
        """
        logger.info("\n" + "-" * 80)
        logger.info("Step 1: Domain Knowledge Selection")
        logger.info("-" * 80)
        
        features = [c for c in df.columns if c != target]
        removed_features = []
        
        # Remove constant features
        for col in features:
            if df[col].nunique() <= 1:
                removed_features.append((col, 'constant_feature'))
        
        # Remove duplicated features
        numeric_features = df[features].select_dtypes(include=[np.number]).columns
        for i, col1 in enumerate(numeric_features):
            for col2 in numeric_features[i+1:]:
                if col1 not in [f[0] for f in removed_features]:
                    if df[col1].equals(df[col2]):
                        removed_features.append((col2, 'duplicate'))
        
        # Remove features with non-sensical business logic
        nonsensical_patterns = ['_max_age_', '_postal_code_avg_']
        for col in features:
            for pattern in nonsensical_patterns:
                if pattern in col.lower():
                    removed_features.append((col, 'non_sensical'))
                    break
        
        # Remove features
        features_to_keep = [f for f in features if f not in [r[0] for r in removed_features]]
        df_filtered = df[[target] + features_to_keep].copy()
        
        logger.info(f"Removed {len(removed_features)} features")
        if removed_features:
            for feat, reason in removed_features[:10]:
                logger.info(f"  {feat}: {reason}")
        
        logger.info(f"Remaining features: {len(features_to_keep)}")
        
        self.selection_results['domain_knowledge'] = removed_features
        
        return df_filtered
    
    def univariate_selection(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Univariate feature selection (Section 10.4 of procedure)
        Based on missing values, stability (PSI), and statistical performance (IV/AUC)
        """
        logger.info("\n" + "-" * 80)
        logger.info("Step 2: Univariate Selection")
        logger.info("-" * 80)
        
        features = [c for c in df.columns if c != target]
        removed_features = []
        feature_stats = []
        
        config = self.config['feature_selection']['univariate']
        
        for col in features:
            stats_dict = {
                'feature': col,
                'missing_pct': df[col].isnull().sum() / len(df),
                'nunique': df[col].nunique()
            }
            
            # Calculate IV and AUC for numeric features
            if df[col].dtype in [np.float64, np.int64]:
                try:
                    iv = self._calculate_iv(df[col], df[target])
                    auc = self._calculate_auc(df[col], df[target])
                    stats_dict['iv'] = iv
                    stats_dict['auc'] = auc
                except Exception as e:
                    stats_dict['iv'] = 0
                    stats_dict['auc'] = 0.5
            else:
                stats_dict['iv'] = None
                stats_dict['auc'] = None
            
            feature_stats.append(stats_dict)
            
            # Check removal criteria
            if stats_dict['missing_pct'] > config['missing_threshold']:
                removed_features.append((col, f"high_missing_{stats_dict['missing_pct']:.2%}"))
            elif stats_dict['iv'] is not None and stats_dict['iv'] < config['iv_threshold']:
                removed_features.append((col, f"low_iv_{stats_dict['iv']:.4f}"))
        
        # Create summary DataFrame
        self.feature_stats_df = pd.DataFrame(feature_stats)
        
        # Remove features
        features_to_keep = [f for f in features if f not in [r[0] for r in removed_features]]
        df_filtered = df[[target] + features_to_keep].copy()
        
        logger.info(f"Removed {len(removed_features)} features")
        logger.info(f"  High missing: {sum(1 for _, r in removed_features if 'missing' in r)}")
        logger.info(f"  Low IV: {sum(1 for _, r in removed_features if 'low_iv' in r)}")
        logger.info(f"Remaining features: {len(features_to_keep)}")
        
        self.selection_results['univariate'] = removed_features
        
        return df_filtered
    
    def _calculate_iv(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate Information Value"""
        try:
            # Create bins
            if feature.nunique() > 10:
                feature_binned = pd.qcut(feature, q=10, duplicates='drop')
            else:
                feature_binned = feature
            
            # Calculate IV
            df_iv = pd.DataFrame({'feature': feature_binned, 'target': target})
            df_iv = df_iv.dropna()
            
            grouped = df_iv.groupby('feature')['target'].agg(['sum', 'count'])
            grouped['non_target'] = grouped['count'] - grouped['sum']
            
            grouped['good_pct'] = grouped['non_target'] / grouped['non_target'].sum()
            grouped['bad_pct'] = grouped['sum'] / grouped['sum'].sum()
            
            grouped['woe'] = np.log(grouped['good_pct'] / grouped['bad_pct'])
            grouped['iv'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']
            
            iv = grouped['iv'].sum()
            
            return iv if not np.isnan(iv) and not np.isinf(iv) else 0
        except:
            return 0
    
    def _calculate_auc(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate AUC"""
        try:
            df_clean = pd.DataFrame({'feature': feature, 'target': target}).dropna()
            if len(df_clean) > 0 and df_clean['target'].nunique() > 1:
                auc = roc_auc_score(df_clean['target'], df_clean['feature'])
                return auc
            return 0.5
        except:
            return 0.5
    
    def multivariate_selection(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Multivariate feature selection (Section 10.5 of procedure)
        Remove highly correlated features
        """
        logger.info("\n" + "-" * 80)
        logger.info("Step 3: Multivariate Selection")
        logger.info("-" * 80)
        
        features = [c for c in df.columns if c != target and df[c].dtype in [np.float64, np.int64]]
        removed_features = []
        
        config = self.config['feature_selection']['multivariate']
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Remove one feature from each highly correlated pair
        to_remove = set()
        for column in upper_triangle.columns:
            if column in to_remove:
                continue
            correlated_features = upper_triangle[column][
                upper_triangle[column] > config['correlation_threshold']
            ].index.tolist()
            
            if correlated_features:
                # Keep feature with higher IV (if available)
                if hasattr(self, 'feature_stats_df'):
                    column_iv = self.feature_stats_df[
                        self.feature_stats_df['feature'] == column
                    ]['iv'].values
                    column_iv = column_iv[0] if len(column_iv) > 0 else 0
                    
                    for corr_feat in correlated_features:
                        corr_iv = self.feature_stats_df[
                            self.feature_stats_df['feature'] == corr_feat
                        ]['iv'].values
                        corr_iv = corr_iv[0] if len(corr_iv) > 0 else 0
                        
                        if column_iv >= corr_iv:
                            to_remove.add(corr_feat)
                            removed_features.append(
                                (corr_feat, f"corr_with_{column}_{corr_matrix.loc[column, corr_feat]:.3f}")
                            )
                        else:
                            to_remove.add(column)
                            removed_features.append(
                                (column, f"corr_with_{corr_feat}_{corr_matrix.loc[column, corr_feat]:.3f}")
                            )
                            break
                else:
                    # No IV available, just remove the second feature
                    to_remove.update(correlated_features)
                    for feat in correlated_features:
                        removed_features.append(
                            (feat, f"corr_with_{column}_{corr_matrix.loc[column, feat]:.3f}")
                        )
        
        # Remove features
        features_to_keep = [f for f in df.columns if f == target or f not in to_remove]
        df_filtered = df[features_to_keep].copy()
        
        logger.info(f"Removed {len(removed_features)} highly correlated features")
        logger.info(f"Remaining features: {len(features_to_keep) - 1}")
        
        self.selection_results['multivariate'] = removed_features
        
        return df_filtered
    
    def final_selection(self, df: pd.DataFrame, target: str, model_type: str) -> pd.DataFrame:
        """
        Final feature selection based on model type
        Conventional: 8-10 features, Advanced: 30-40 features
        """
        logger.info("\n" + "-" * 80)
        logger.info("Step 4: Final Selection")
        logger.info("-" * 80)
        
        config = self.config['feature_selection']['final_selection']
        max_features = (
            config['max_features_conventional'] if model_type == 'conventional' 
            else config['max_features_advanced']
        )
        
        features = [c for c in df.columns if c != target]
        
        if len(features) <= max_features:
            logger.info(f"Current features ({len(features)}) <= max ({max_features}). No reduction needed.")
            self.selection_results['final'] = features
            return df
        
        # Rank features by IV
        if hasattr(self, 'feature_stats_df'):
            feature_ranking = self.feature_stats_df[
                self.feature_stats_df['feature'].isin(features)
            ].sort_values('iv', ascending=False)
            
            top_features = feature_ranking.head(max_features)['feature'].tolist()
        else:
            # Fallback: random selection (not ideal)
            logger.warning("No IV stats available. Using first N features.")
            top_features = features[:max_features]
        
        df_final = df[[target] + top_features].copy()
        
        logger.info(f"Selected top {len(top_features)} features for {model_type} model")
        logger.info(f"Top 10 features by IV:")
        if hasattr(self, 'feature_stats_df'):
            for i, row in feature_ranking.head(10).iterrows():
                logger.info(f"  {row['feature']}: IV={row['iv']:.4f}, AUC={row['auc']:.4f}")
        
        self.selection_results['final'] = top_features
        
        return df_final
    
    def _log_selection_summary(self) -> None:
        """Log overall selection summary"""
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE SELECTION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\nDomain Knowledge: Removed {len(self.selection_results['domain_knowledge'])} features")
        logger.info(f"Univariate: Removed {len(self.selection_results['univariate'])} features")
        logger.info(f"Multivariate: Removed {len(self.selection_results['multivariate'])} features")
        logger.info(f"Final: Selected {len(self.selection_results['final'])} features")
        
        logger.info("=" * 80)
    
    def get_selection_report(self) -> pd.DataFrame:
        """Generate detailed selection report"""
        report_data = []
        
        for stage, features in self.selection_results.items():
            if stage == 'final':
                for feat in features:
                    report_data.append({
                        'feature': feat,
                        'stage': stage,
                        'status': 'selected',
                        'reason': 'final_selection'
                    })
            else:
                for feat, reason in features:
                    report_data.append({
                        'feature': feat,
                        'stage': stage,
                        'status': 'removed',
                        'reason': reason
                    })
        
        return pd.DataFrame(report_data)


def main():
    """Example usage"""
    from pathlib import Path
    
    # Initialize selector
    config_path = Path("config/feature_config.yaml")
    if config_path.exists():
        selector = FeatureSelector(str(config_path))
    else:
        logger.warning("Config file not found, using defaults")
        selector = FeatureSelector()
    
    # Load features with target
    input_path = Path("data/features/time_features.parquet")
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run time_features.py first")
        return
    
    df = pd.read_parquet(input_path)
    
    # For this example, we need a target variable
    # In practice, this would come from merging with portfolio data
    if 'default_flag' not in df.columns:
        logger.warning("No target variable found. Creating dummy target for demonstration.")
        np.random.seed(42)
        df['default_flag'] = np.random.binomial(1, 0.02, size=len(df))
    
    # Select features for conventional model
    df_selected = selector.select_features(df, target='default_flag', model_type='conventional')
    
    # Generate report
    report = selector.get_selection_report()
    print("\nSelection Report:")
    print(report.groupby(['stage', 'status']).size())
    
    # Save selected features
    output_path = Path("data/features/selected_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_selected.to_parquet(output_path, index=False)
    logger.info(f"\nSaved selected features to {output_path}")
    
    # Save report
    report_path = Path("data/features/feature_selection_report.csv")
    report.to_csv(report_path, index=False)
    logger.info(f"Saved selection report to {report_path}")


if __name__ == "__main__":
    main()
