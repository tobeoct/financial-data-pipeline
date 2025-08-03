# Assignment #3: Hands-On Data Prep

## Overview
This assignment demonstrates comprehensive data preparation using a financial dataset with pandas. The implementation follows professional coding practices with configuration management, strong typing, and modular design.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the complete data preparation pipeline:
```bash
python main.py
```

## Assignment Components

### 1. Data Exploration (2 points)
- **File:** `data_explorer.py`
- Thorough analysis of dataset characteristics, patterns, and quality
- Statistical summaries, missing value analysis, outlier detection
- Correlation analysis and target variable distribution

### 2. Data Cleansing (3 points)
- **File:** `data_cleaner.py`
- Effective handling of missing values, duplicates, and inconsistencies
- Advanced imputation strategies using sector-based grouping
- Outlier detection and handling using statistical methods

### 3. Feature Engineering (3 points)
- **File:** `feature_engineer.py`
- Creative and meaningful feature creation and transformation
- Financial ratios, temporal features, polynomial transformations
- Categorical binning, sector-based features, composite scores

### 4. Code Implementation (2 points)
- **File:** `dimensionality_reducer.py`, configuration system
- Proper use of Pandas with efficient and readable code
- Strong typing, configuration management, modular design
- Dimensionality reduction with PCA and feature selection

## Dataset
Financial dataset with realistic market data including:
- Stock prices, volumes, market capitalization
- Financial ratios (PE, PB, debt-to-equity, ROA)
- Sector classifications and temporal data
- Investment grade classifications as target variable

## Configuration
All parameters are externalized in `config.yaml`:
- Sample sizes, random seeds
- Data quality issue rates
- Processing thresholds and strategies

## Output
- `final_dataset.csv` - Final processed dataset
- Console output showing detailed processing steps and results