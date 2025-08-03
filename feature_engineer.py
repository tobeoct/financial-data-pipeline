"""
Feature Engineer  - Feature Engineering (3 points)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_types import Dataset, Config


class FeatureEngineer:
    """Creative and meaningful feature creation and transformation"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def engineer_features(self, dataset: Dataset) -> Dataset:
        """Comprehensive feature engineering pipeline"""
        df = dataset.data.copy()
        
        print("=" * 50)
        print("FEATURE ENGINEERING")
        print("=" * 50)
        
        print(f"Starting with {df.shape[1]} features")
        
        # 1. Temporal features from date
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.dayofweek
            print("Created temporal features: year, month, quarter, day_of_week")
        
        # 2. Financial ratio features
        # Price-to-volume ratio
        df['price_volume_ratio'] = df['close_price'] / (df['volume'] + 1)
        
        # Market efficiency ratios
        df['price_revenue_ratio'] = df['close_price'] / (df['revenue'] / 1e6 + 1)  # Price per million revenue
        df['market_cap_revenue_ratio'] = df['market_cap'] / (df['revenue'] + 1)
        
        # Risk indicators
        df['return_volatility_ratio'] = df['daily_return'] / (df['volatility'] + 0.001)
        df['debt_equity_normalized'] = df['debt_to_equity'] / (df['debt_to_equity'].median())
        
        print("Created financial ratio features")
        
        # 3. Polynomial features for key variables
        degree = self.config.polynomial_degree
        key_features = ['close_price', 'market_cap', 'pe_ratio']
        
        for feature in key_features:
            if feature in df.columns:
                df[f'{feature}_squared'] = df[feature] ** degree
                df[f'{feature}_log'] = np.log1p(df[feature])  # log(1+x) to handle zeros
        
        print(f"Created polynomial and log transformation features")
        
        # 4. Binning continuous variables
        # Market cap categories
        df['market_cap_category'] = pd.cut(df['market_cap'], 
                                         bins=self.config.n_bins, 
                                         labels=['Micro', 'Small', 'Mid', 'Large', 'Mega'])
        
        # Performance categories
        df['return_category'] = pd.cut(df['daily_return'],
                                     bins=[-np.inf, -0.02, 0, 0.02, np.inf],
                                     labels=['Poor', 'Below_Avg', 'Average', 'Good'])
        
        # PE ratio categories
        df['pe_category'] = pd.cut(df['pe_ratio'],
                                 bins=[0, 15, 25, 35, np.inf],
                                 labels=['Low', 'Moderate', 'High', 'Very_High'])
        
        print("Created categorical binning features")
        
        # 5. Sector-based features
        sector_stats = df.groupby('sector').agg({
            'close_price': ['mean', 'std'],
            'market_cap': 'mean',
            'pe_ratio': 'median'
        }).round(3)
        
        sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns]
        sector_stats = sector_stats.add_prefix('sector_')
        
        # Merge sector statistics
        df = df.merge(sector_stats, left_on='sector', right_index=True, how='left')
        
        # Calculate relative position within sector
        df['price_vs_sector'] = df['close_price'] / df['sector_close_price_mean']
        df['market_cap_vs_sector'] = df['market_cap'] / df['sector_market_cap_mean']
        
        print("Created sector-based comparative features")
        
        # 6. Interaction features
        # High-value interactions
        df['pe_pb_interaction'] = df['pe_ratio'] * df['pb_ratio']
        df['size_performance'] = df['market_cap'] * df['daily_return']
        df['risk_return'] = df['volatility'] * df['daily_return']
        
        print("Created interaction features")
        
        # 7. Encode categorical variables
        categorical_cols = ['sector', 'market_cap_category', 'return_category', 'pe_category']
        
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == 'object':
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        print("Encoded categorical variables")
        
        # 8. Create composite financial health score
        # Normalize key metrics to 0-1 scale
        metrics = ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'roa']
        normalized_metrics = {}
        
        for metric in metrics:
            if metric in df.columns:
                min_val = df[metric].min()
                max_val = df[metric].max()
                normalized_metrics[f'{metric}_norm'] = (df[metric] - min_val) / (max_val - min_val)
        
        # Composite score (lower debt and PE is better, higher ROA is better)
        if len(normalized_metrics) >= 3:
            df['financial_health_score'] = (
                (1 - normalized_metrics.get('debt_to_equity_norm', 0)) * 0.3 +
                (1 - normalized_metrics.get('pe_ratio_norm', 0)) * 0.3 +
                normalized_metrics.get('roa_norm', 0) * 0.4
            )
            print("Created composite financial health score")
        
        print(f"Feature engineering completed: {df.shape[1]} features")
        
        return Dataset(
            data=df,
            target_column=dataset.target_column,
            feature_columns=[col for col in df.columns if col != dataset.target_column],
            metadata=dataset.metadata
        )