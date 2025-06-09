"""
Data Loader Module for Neural Network Project
Handles all data loading, preprocessing, and splitting operations
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

class DataLoader:
    """
    Comprehensive data loader with preprocessing capabilities
    """
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.feature_names = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file with configuration parameters"""
        data_path = self.config.get_data_path()
        
        # Get loading parameters
        total_rows = self.config.get('data.loading.total_rows')
        encoding = self.config.get('data.encoding', 'utf-8')
        separator = self.config.get('data.separator')
        
        self.logger.info(f"Loading data from: {data_path}")
        self.logger.info(f"Max rows to load: {total_rows}")
        
        try:
            # Load data with parameters
            data = pd.read_csv(
                data_path,
                sep=separator,
                engine='python',
                nrows=total_rows,
                encoding=encoding
            )
            
            self.logger.info(f"Data loaded successfully: {data.shape}")
            self.logger.info(f"Columns: {list(data.columns)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive preprocessing pipeline"""
        self.logger.info("Starting data preprocessing...")
        original_shape = data.shape
        
        # Step 1: Drop configured columns
        data = self._drop_configured_columns(data)
        
        # Step 2: Handle data types
        data = self._convert_data_types(data)
        
        # Step 3: Handle missing values
        data = self._handle_missing_values(data)
        
        # Step 4: Handle outliers
        if self.config.get('data.preprocessing.handle_outliers', False):
            data = self._handle_outliers(data)
        
        # Step 5: Remove remaining non-numeric columns
        data = self._ensure_numeric_data(data)
        
        self.logger.info(f"Preprocessing complete: {original_shape} -> {data.shape}")
        return data
    
    def _drop_configured_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop columns specified in configuration"""
        columns_to_drop = self.config.get('data.columns_to_drop', [])
        existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]
        
        if existing_cols_to_drop:
            data = data.drop(columns=existing_cols_to_drop)
            self.logger.info(f"Dropped {len(existing_cols_to_drop)} columns from config")
            
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for better processing"""
        # Convert comma decimals to proper float format
        for col in data.select_dtypes(include=['object']).columns:
            try:
                if data[col].astype(str).str.contains(',').any():
                    self.logger.debug(f"Converting comma decimals in column: {col}")
                    data[col] = data[col].astype(str).str.replace(',', '.').replace('nan', np.nan)
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            except:
                pass
        
        # Convert boolean columns to integers
        bool_cols = data.select_dtypes(include=['bool']).columns.tolist()
        for col in bool_cols:
            data[col] = data[col].astype(int)
            self.logger.debug(f"Converted boolean column to int: {col}")
        
        # Convert yes/no columns to numeric
        for col in data.select_dtypes(include=['object']).columns:
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) <= 2 and all(val.lower() in ['yes', 'no'] for val in unique_vals if isinstance(val, str)):
                self.logger.debug(f"Converting yes/no column to numeric: {col}")
                data[col] = data[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0})
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        missing_threshold = self.config.get('data.missing_data.threshold', 0.7)
        strategy = self.config.get('data.missing_data.strategy', 'median')
        
        # Drop columns with too many missing values
        missing_ratios = data.isnull().sum() / len(data)
        cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index.tolist()
        
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            self.logger.info(f"Dropped {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
        
        # Fill remaining missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            if strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            else:
                imputer = SimpleImputer(strategy='constant', fill_value=0)
            
            data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
            self.logger.info(f"Filled missing values using {strategy} strategy")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove price outliers (extreme prices only)"""
        
        # Handle price outliers separately (remove rows with extreme prices)
        if 'price' in data.columns:
            initial_rows = len(data)
            # Remove houses above 1,000,000 zł and below 50,000 zł
            data = data[(data['price'] >= 50000) & (data['price'] <= 1000000)]
            removed_rows = initial_rows - len(data)
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} rows with extreme prices (outside 50k-1M zł range)")
        
        return data
    
    def _ensure_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all remaining columns are numeric"""
        non_numeric_cols = data.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            self.logger.warning(f"Removing remaining non-numeric columns: {non_numeric_cols}")
            data = data.drop(columns=non_numeric_cols)
        
        return data
    
    def split_features_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and target"""
        if 'price' in data.columns:
            X = data.drop(columns=['price'])
            y = data['price']
            self.feature_names = X.columns.tolist()
            self.logger.info(f"Split data: {X.shape[1]} features, {X.shape[0]} samples")
            return X, y
        else:
            # For data without price column (e.g., prediction data)
            self.feature_names = data.columns.tolist()
            self.logger.info(f"Feature data: {data.shape[1]} features, {data.shape[0]} samples")
            return data, None
    
    def create_data_splits(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create train/validation/test splits with proper shuffling"""
        # Get split ratios
        train_ratio = self.config.get('data.loading.train_ratio', 0.7)
        val_ratio = self.config.get('data.loading.val_ratio', 0.15)
        test_ratio = self.config.get('data.loading.test_ratio', 0.15)
        
        shuffle_data = self.config.get('data.loading.shuffle_data', True)
        random_state = self.config.get('data.loading.random_state', 42)
        
        self.logger.info(f"Creating data splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # First split: separate test set
        test_size = test_ratio
        train_val_size = train_ratio + val_ratio
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle_data
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_ratio / train_val_size
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            shuffle=shuffle_data
        )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val, 
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        # Log split sizes
        for split_name, split_data in splits.items():
            self.logger.info(f"{split_name}: {len(split_data)} samples")
        
        return splits
    
    def normalize_features(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize features using configured scaler"""
        if not self.config.get('data.preprocessing.normalize_features', True):
            return data_splits
        
        scaler_type = self.config.get('data.preprocessing.scaler_type', 'standard')
        
        # Initialize scaler
        if scaler_type == 'standard':
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif scaler_type == 'robust':
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
        elif scaler_type == 'minmax':
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        else:
            self.logger.warning(f"Unknown scaler type: {scaler_type}. Using StandardScaler.")
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        
        # Fit scalers on training data
        X_train_scaled = scaler_X.fit_transform(data_splits['X_train'])
        y_train_scaled = scaler_y.fit_transform(data_splits['y_train'].values.reshape(-1, 1)).ravel()
        
        # Transform all splits
        X_val_scaled = scaler_X.transform(data_splits['X_val'])
        X_test_scaled = scaler_X.transform(data_splits['X_test'])
        
        y_val_scaled = scaler_y.transform(data_splits['y_val'].values.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(data_splits['y_test'].values.reshape(-1, 1)).ravel()
        
        # Store scalers for later use
        self.scalers = {
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
        
        # Update data splits
        normalized_splits = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_val': y_val_scaled,
            'y_test': y_test_scaled,
            'scalers': self.scalers
        }
        
        self.logger.info(f"Features normalized using {scaler_type} scaler")
        return normalized_splits
    
    def prepare_data_pipeline(self) -> Dict[str, Any]:
        """Complete data preparation pipeline"""
        # Load data
        raw_data = self.load_data()
        
        # Preprocess data
        processed_data = self.preprocess_data(raw_data)
        
        # Split features and target
        X, y = self.split_features_target(processed_data)
        
        # Create train/val/test splits
        data_splits = self.create_data_splits(X, y)
        
        # Normalize features
        normalized_splits = self.normalize_features(data_splits)
        
        # Add metadata
        normalized_splits['feature_names'] = self.feature_names
        normalized_splits['original_shape'] = raw_data.shape
        normalized_splits['processed_shape'] = processed_data.shape
        
        self.logger.info("Data pipeline completed successfully")
        return normalized_splits
