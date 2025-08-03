"""
Dimensionality Reducer  - Code Implementation Enhancement
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from data_types import Dataset, Config


class DimensionalityReducer:
    """Reduces feature dimensionality while maintaining information"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def reduce_dimensions(self, dataset: Dataset) -> Dataset:
        """Apply dimensionality reduction techniques"""
        df = dataset.data.copy()
        
        print("=" * 50)
        print("DIMENSIONALITY REDUCTION")
        print("=" * 50)
        
        # Prepare features
        numeric_features = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_features if col != dataset.target_column]
        
        X = df[feature_cols]
        y = df[dataset.target_column]
        
        print(f"Starting with {len(feature_cols)} features")
        
        # 1. Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.95
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > 0.95)]
        
        X_filtered = X.drop(columns=to_drop)
        print(f"Removed {len(to_drop)} highly correlated features")
        
        # 2. Feature selection using Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
        rf.fit(X_filtered, y)
        
        feature_importance = pd.DataFrame({
            'feature': X_filtered.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        n_features = min(self.config.n_features, len(X_filtered.columns))
        top_features = feature_importance.head(n_features)['feature'].tolist()
        X_selected = X_filtered[top_features]
        
        print(f"Selected top {len(top_features)} features by importance")
        print("Top 10 features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # 3. Apply PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        pca = PCA(n_components=self.config.pca_variance, random_state=self.config.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA feature names
        pca_features = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_features, index=df.index)
        
        print(f"PCA reduced to {X_pca.shape[1]} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Combine with non-numeric features and target
        non_numeric_cols = [col for col in df.columns 
                           if col not in numeric_features and col != dataset.target_column]
        
        final_df = pd.concat([
            df[['company_id'] + non_numeric_cols],  # Keep ID and categorical
            X_pca_df,  # PCA features
            df[[dataset.target_column]]  # Target
        ], axis=1)
        
        print(f"Final dataset: {final_df.shape[1]} features")
        print(f"Reduction: {len(feature_cols)} -> {X_pca.shape[1]} features ({(1-X_pca.shape[1]/len(feature_cols))*100:.1f}% reduction)")
        
        return Dataset(
            data=final_df,
            target_column=dataset.target_column,
            feature_columns=[col for col in final_df.columns if col != dataset.target_column],
            metadata={
                **dataset.metadata,
                'dimensionality_reduction': {
                    'original_features': len(feature_cols),
                    'final_features': X_pca.shape[1],
                    'explained_variance': pca.explained_variance_ratio_.sum(),
                    'top_features': top_features[:10]
                }
            }
        )