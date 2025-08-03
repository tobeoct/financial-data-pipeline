"""
Data Prep:
Use a real dataset to perform data cleansing and feature engineering using Pandas
"""

from config_loader import load_config
from data_loader import DataLoader
from data_explorer import DataExplorer
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from dimensionality_reducer import DimensionalityReducer


def main():
    """Execute Data Preparation Pipeline"""
    
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_loader = DataLoader(config)
    data_explorer = DataExplorer()
    data_cleaner = DataCleaner(config)
    feature_engineer = FeatureEngineer(config)
    dimensionality_reducer = DimensionalityReducer(config)
    
    # 1. Load financial dataset
    print("\nLoading financial dataset...")
    dataset = data_loader.load_data()
    print(f"Loaded dataset with {dataset.data.shape[0]} records and {dataset.data.shape[1]} features")
    
    # 2. Data Exploration
    exploration_results = data_explorer.explore_dataset(dataset)
    
    # 3. Data Cleansing 
    clean_dataset = data_cleaner.clean_data(dataset)
    
    # 4. Feature Engineering 
    engineered_dataset = feature_engineer.engineer_features(clean_dataset)
    
    # 5. Dimensionality Reduction (Code Implementation enhancement)
    final_dataset = dimensionality_reducer.reduce_dimensions(engineered_dataset)
    
    # Save results
    final_dataset.data.to_csv('final_dataset.csv', index=False)
    
    # Summary
    print("\n" + "=" * 50)
    print("COMPLETION SUMMARY")
    print("=" * 50)
    print(f"COMPLETED Data Exploration: Analyzed {dataset.data.shape[0]} records")
    print(f"COMPLETED Data Cleansing: Handled missing values, duplicates, outliers")
    print(f"COMPLETED Feature Engineering: Created {engineered_dataset.data.shape[1] - dataset.data.shape[1]} new features")
    print(f"COMPLETED Code Implementation: Efficient pandas operations with {final_dataset.data.shape[1]} final features")
    print(f"COMPLETED Final dataset saved as: final_dataset.csv")
    print("=" * 50)
    
    return final_dataset


if __name__ == "__main__":
    final_dataset = main()