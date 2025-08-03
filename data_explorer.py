"""
Data Explorer  - Data Exploration
"""

import pandas as pd
import numpy as np
from data_types import Dataset


class DataExplorer:
    """Performs thorough analysis of dataset characteristics, patterns, and quality"""
    
    def explore_dataset(self, dataset: Dataset) -> dict:
        """Comprehensive data exploration"""
        df = dataset.data
        
        print("=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)
        
        # Basic dataset information
        print(f"\nBasic Information:")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Data types analysis
        print(f"\nData Types:")
        print(df.dtypes.value_counts())
        
        # Missing values analysis
        print(f"\nMissing Values:")
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_stats,
            'Missing_Percentage': missing_percent
        }).round(2)
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Duplicate analysis
        duplicates = df.duplicated().sum()
        print(f"\nDuplicates: {duplicates} ({(duplicates/len(df)*100):.1f}%)")
        
        # Statistical summary
        print(f"\nStatistical Summary:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe().round(2))
        
        # Categorical analysis
        print(f"\nCategorical Variables:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'company_id':  # Skip ID column
                print(f"\n{col}:")
                print(df[col].value_counts().head())
        
        # Outlier detection
        print(f"\nOutlier Analysis (IQR method):")
        outlier_summary = {}
        for col in numeric_cols:
            if col != dataset.target_column:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
                if outliers > 0:
                    outlier_summary[col] = outliers
        
        for col, count in outlier_summary.items():
            print(f"{col}: {count} outliers")
        
        # Correlation analysis
        print(f"\nCorrelation Analysis:")
        corr_matrix = df[numeric_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print("High correlations (>0.7):")
            for col1, col2, corr_val in high_corr_pairs:
                print(f"  {col1} - {col2}: {corr_val:.3f}")
        else:
            print("No high correlations found")
        
        # Target variable analysis
        print(f"\nTarget Variable Analysis:")
        target_counts = df[dataset.target_column].value_counts().sort_index()
        print(f"Investment Grade Distribution:")
        for grade, count in target_counts.items():
            print(f"  Grade {grade}: {count} ({count/len(df)*100:.1f}%)")
        
        return {
            'shape': df.shape,
            'missing_values': missing_stats.sum(),
            'duplicates': duplicates,
            'outliers': sum(outlier_summary.values()),
            'high_correlations': len(high_corr_pairs),
            'target_distribution': target_counts.to_dict()
        }