"""
Data Cleaner for Assignment #3 - Data Cleansing (3 points)
"""

import pandas as pd
import numpy as np
from data_types import Dataset, Config


class DataCleaner:
    """Effective handling of missing values, duplicates, and inconsistencies"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def clean_data(self, dataset: Dataset) -> Dataset:
        """Comprehensive data cleaning pipeline"""
        df = dataset.data.copy()
        
        print("=" * 50)
        print("DATA CLEANSING")
        print("=" * 50)
        
        print(f"Starting with {len(df)} records")
        
        # Remove duplicates
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"Removed {duplicates_before} duplicate records")
        
        # Handle invalid values
        # Negative prices are impossible
        invalid_prices = (df['close_price'] < 0).sum()
        if invalid_prices > 0:
            df.loc[df['close_price'] < 0, 'close_price'] = np.nan
            print(f"Set {invalid_prices} negative prices to NaN")
        
        # Zero volumes are problematic
        zero_volumes = (df['volume'] == 0).sum()
        if zero_volumes > 0:
            df.loc[df['volume'] == 0, 'volume'] = np.nan
            print(f"Set {zero_volumes} zero volumes to NaN")
        
        # Handle extreme outliers (>3 standard deviations)
        numeric_cols = ['close_price', 'volume', 'pe_ratio', 'market_cap']
        outliers_handled = 0
        
        for col in numeric_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                threshold = self.config.outlier_threshold
                
                outlier_mask = np.abs(df[col] - mean_val) > threshold * std_val
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    df.loc[outlier_mask, col] = np.nan
                    outliers_handled += outlier_count
        
        print(f"Set {outliers_handled} extreme outliers to NaN")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Strategy: Use median imputation grouped by sector
        numeric_cols_to_impute = ['close_price', 'volume', 'pe_ratio', 'pb_ratio', 
                                'market_cap', 'revenue', 'debt_to_equity', 'roa']
        
        for col in numeric_cols_to_impute:
            if col in df.columns and df[col].isnull().any():
                # Group by sector and impute with median
                df[col] = df.groupby('sector')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill remaining NaN with overall median
                df[col] = df[col].fillna(df[col].median())
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values reduced from {missing_before} to {missing_after}")
        
        # Standardize categorical data
        if 'sector' in df.columns:
            # Ensure consistent formatting
            df['sector'] = df['sector'].str.title().str.strip()
            print(f"Standardized sector names")
        
        # Data consistency checks
        print(f"Final dataset: {len(df)} records, {df.shape[1]} features")
        
        return Dataset(
            data=df,
            target_column=dataset.target_column,
            feature_columns=[col for col in df.columns if col != dataset.target_column],
            metadata=dataset.metadata
        )