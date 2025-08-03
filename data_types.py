"""
Data types and models 
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class Config:
    """Configuration for data processing"""
    sample_size: int
    random_state: int
    missing_value_rate: float
    duplicate_rate: float
    outlier_rate: float
    outlier_threshold: float
    imputation_strategy: str
    polynomial_degree: int
    n_bins: int
    n_features: int
    pca_variance: float


@dataclass
class Dataset:
    """Container for dataset with metadata"""
    data: pd.DataFrame
    target_column: str
    feature_columns: List[str]
    metadata: Dict[str, Any]