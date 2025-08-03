"""
Data Loader  - Financial Dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from data_types import Dataset, Config


class DataLoader:
    """Loads financial dataset for data preparation"""
    
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.random_state)
    
    def create_financial_dataset(self) -> Dataset:
        """Create realistic financial dataset"""
        n_samples = self.config.sample_size
        
        # Financial sectors
        sectors = ['Technology', 'Healthcare', 'Financial', 'Consumer_Discretionary',
                  'Industrials', 'Energy', 'Materials', 'Utilities', 'Real_Estate']
        
        # Generate financial data
        data = {
            'company_id': [f"COMP_{i:04d}" for i in range(1, n_samples + 1)],
            'sector': np.random.choice(sectors, size=n_samples),
            'close_price': np.random.lognormal(mean=3.0, sigma=1.2, size=n_samples),
            'volume': np.random.lognormal(mean=12.0, sigma=1.5, size=n_samples).astype(int),
            'market_cap': np.random.lognormal(mean=18.0, sigma=2.0, size=n_samples),
            'pe_ratio': np.random.gamma(shape=2.0, scale=12.0, size=n_samples),
            'pb_ratio': np.random.gamma(shape=1.5, scale=1.8, size=n_samples),
            'daily_return': np.random.normal(loc=0.0005, scale=0.02, size=n_samples),
            'volatility': np.random.gamma(shape=2.0, scale=0.15, size=n_samples),
            'revenue': np.random.lognormal(mean=16.0, sigma=1.5, size=n_samples),
            'debt_to_equity': np.random.gamma(shape=1.2, scale=0.5, size=n_samples),
            'roa': np.random.normal(loc=0.08, scale=0.05, size=n_samples)
        }
        
        # Add date
        start_date = datetime(2020, 1, 1)
        data['date'] = [start_date + timedelta(days=np.random.randint(0, 1460)) for _ in range(n_samples)]
        
        df = pd.DataFrame(data)
        
        # Create target variable (investment grade)
        sector_scores = {'Technology': 0.75, 'Healthcare': 0.70, 'Financial': 0.60,
                        'Consumer_Discretionary': 0.55, 'Industrials': 0.50, 'Energy': 0.45,
                        'Materials': 0.48, 'Utilities': 0.65, 'Real_Estate': 0.52}
        
        base_score = df['sector'].map(sector_scores)
        market_cap_score = (df['market_cap'] - df['market_cap'].min()) / (df['market_cap'].max() - df['market_cap'].min())
        return_score = (df['daily_return'] - df['daily_return'].min()) / (df['daily_return'].max() - df['daily_return'].min())
        
        investment_score = (base_score * 0.5 + market_cap_score * 0.3 + return_score * 0.2 +
                          np.random.normal(0, 0.1, len(df)))
        
        df['investment_grade'] = pd.cut(investment_score, bins=4, labels=[0, 1, 2, 3]).astype(int)
        
        return Dataset(
            data=df,
            target_column='investment_grade',
            feature_columns=[col for col in df.columns if col != 'investment_grade'],
            metadata={'sectors': sectors, 'sample_size': n_samples}
        )
    
    def add_data_issues(self, dataset: Dataset) -> Dataset:
        """Add realistic data quality issues"""
        df = dataset.data.copy()
        
        # Add missing values
        n_missing = int(self.config.missing_value_rate * len(df))
        missing_cols = ['close_price', 'volume', 'pe_ratio', 'revenue']
        
        for col in missing_cols:
            missing_indices = np.random.choice(df.index, size=n_missing//len(missing_cols), replace=False)
            df.loc[missing_indices, col] = np.nan
        
        # Add duplicates
        n_duplicates = int(self.config.duplicate_rate * len(df))
        duplicate_indices = np.random.choice(df.index, size=n_duplicates, replace=False)
        duplicate_rows = df.loc[duplicate_indices].copy()
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        
        # Add outliers
        n_outliers = int(self.config.outlier_rate * len(df))
        outlier_indices = np.random.choice(df.index, size=n_outliers, replace=False)
        
        half_outliers = n_outliers // 2
        price_indices = outlier_indices[:half_outliers]
        volume_indices = outlier_indices[half_outliers:half_outliers*2]
        
        df.loc[price_indices, 'close_price'] *= np.random.uniform(5, 20, len(price_indices))
        df.loc[volume_indices, 'volume'] *= np.random.uniform(50, 200, len(volume_indices))
        
        return Dataset(
            data=df,
            target_column=dataset.target_column,
            feature_columns=dataset.feature_columns,
            metadata=dataset.metadata
        )
    
    def load_data(self) -> Dataset:
        """Load complete dataset with data quality issues"""
        clean_dataset = self.create_financial_dataset()
        return self.add_data_issues(clean_dataset)