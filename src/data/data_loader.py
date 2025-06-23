"""
Data Loader Module for Neural Network Project
Handles all data loading, preprocessing, and splitting operations
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


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
        total_rows = self.config.get("data.loading.total_rows")
        encoding = self.config.get("data.encoding", "utf-8")
        separator = self.config.get("data.separator")

        self.logger.info(f"Loading data from: {data_path}")
        self.logger.info(f"Max rows to load: {total_rows}")

        try:
            # Użyj chunksize dla pokazania postępu
            chunks = []
            chunk_size = 1000  # możesz dostosować rozmiar chunka

            with tqdm(total=total_rows, desc="Loading data") as pbar:
                for chunk in pd.read_csv(
                        data_path,
                        sep=separator,
                        engine="python",
                        nrows=total_rows,
                        encoding=encoding,
                        chunksize=chunk_size
                ):
                    chunks.append(chunk)
                    pbar.update(len(chunk))

            data = pd.concat(chunks, ignore_index=True)

            self.logger.info(f"Data loaded successfully: {data.shape}")
            self.logger.info(f"Columns: {list(data.columns)}")

            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _encode_categorical_columns(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode categorical columns using one-hot or embedding index encoding.
        """
        # Define which columns to embed (high-cardinality)
        embedding_cols = self.config.get(
            "data.categorical.embedding_columns", ["adress"]
        )
        # Define which columns to one-hot (low-cardinality)
        onehot_cols = self.config.get(
            "data.categorical.onehot_columns",
            ["city", "property_type", "ownership_type", "voivodeship"],
        )
        embedding_info = {}

        # Handle embedding columns: convert to category codes
        for col in embedding_cols:
            if col in data.columns:
                data[col] = data[col].astype("category")
                data[col + "_idx"] = data[col].cat.codes
                embedding_info[col] = {
                    "num_embeddings": data[col].nunique(),
                    "col_idx": data.columns.get_loc(col + "_idx"),
                }
                data = data.drop(columns=[col])

        # Handle one-hot columns
        onehot_present = [col for col in onehot_cols if col in data.columns]
        data = pd.get_dummies(data, columns=onehot_present, drop_first=True)

        return data, embedding_info

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply comprehensive preprocessing pipeline,
        including categorical encoding.
        """
        self.logger.info("Starting data preprocessing...")
        original_shape = data.shape

        # Lista kroków preprocessingu
        preprocessing_steps = [
            ("Selecting configured columns", self._select_configured_columns),
            ("Converting data types", self._convert_data_types),
            ("Handling missing values", self._handle_missing_values),
            ("Handling outliers",
             lambda x: self._handle_outliers(x) if self.config.get("data.preprocessing.handle_outliers", False) else x),
            ("Encoding categorical columns", lambda x: self._encode_categorical_columns(x)[0]),
            # bierzemy tylko pierwszy element z tuple
            ("Ensuring numeric data", self._ensure_numeric_data)
        ]

        # Utworzenie paska postępu
        with tqdm(total=len(preprocessing_steps), desc="Preprocessing data") as pbar:
            for step_name, step_func in preprocessing_steps:
                pbar.set_description(f"Preprocessing: {step_name}")
                data = step_func(data)
                pbar.update(1)

        # Get embedding info separately since we modified the encoding step above
        _, embedding_info = self._encode_categorical_columns(data)

        # Log final columns used for training
        self._log_training_columns(data)

        self.logger.info("Preprocessing complete: %s -> %s", original_shape, data.shape)
        return data, embedding_info

    # def _drop_configured_columns(self, data: pd.DataFrame) -> pd.DataFrame:
    #     """Drop columns specified in configuration"""
    #     columns_to_drop = self.config.get("data.columns_to_drop", [])
    #     existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]
    #     if existing_cols_to_drop:
    #         self.logger.info(
    #             "Dropped %d columns from config", len(existing_cols_to_drop)
    #         )
    #         data = data.drop(columns=existing_cols_to_drop)
    #     return data

    def _select_configured_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select columns specified in configuration"""
        columns_to_add = self.config.get('data.columns_to_add', [])
        existing_cols_to_add = [col for col in columns_to_add if col in data.columns]

        if existing_cols_to_add:
            data = data[existing_cols_to_add]
            self.logger.info(f"Selected {len(existing_cols_to_add)} columns from config")

        return data

    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types for better processing"""
        # Convert comma decimals to proper float format
        for col in data.select_dtypes(include=["object"]).columns:
            try:
                if data[col].astype(str).str.contains(",").any():
                    self.logger.debug("Converting comma decimals in column: %s", col)
                    data[col] = data[col].astype(str).str.replace(",", ".")
                    data[col] = data[col].replace("nan", np.nan)
                    data[col] = pd.to_numeric(data[col], errors="coerce")
            except Exception:
                pass

        # Convert boolean columns to integers
        bool_cols = data.select_dtypes(include=["bool"]).columns.tolist()
        for col in bool_cols:
            data[col] = data[col].astype(int)
            self.logger.debug(f"Converted boolean column to int: {col}")

        # Convert yes/no columns to numeric
        for col in data.select_dtypes(include=["object"]).columns:
            unique_vals = data[col].dropna().unique()
            if len(unique_vals) <= 2 and all(
                isinstance(val, str) and val.lower() in ["yes", "no"]
                for val in unique_vals
            ):
                self.logger.debug("Converting yes/no column to numeric: %s", col)
                data[col] = data[col].map(
                    {"yes": 1, "no": 0, "Yes": 1, "No": 0, "YES": 1, "NO": 0}
                )

        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        missing_threshold = self.config.get("data.missing_data.threshold", 0.7)
        strategy = self.config.get("data.missing_data.strategy", "median")

        # Drop columns with too many missing values
        missing_ratios = data.isnull().sum() / len(data)
        cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index.tolist()

        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            self.logger.info(
                f"Dropped {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values"
            )

        # Fill remaining missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) > 0:
            if strategy == "median":
                imputer = SimpleImputer(strategy="median")
            elif strategy == "mean":
                imputer = SimpleImputer(strategy="mean")
            else:
                imputer = SimpleImputer(strategy="constant", fill_value=0)

            data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
            self.logger.info(f"Filled missing values using {strategy} strategy")

        return data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove price outliers and unrealistic room counts"""

        initial_rows = len(data)

        # Handle price outliers separately (remove rows with extreme prices)
        if "price" in data.columns:
            # Remove houses above 1,000,000 zł and below 50,000 zł
            data = data[(data["price"] >= 50000) & (data["price"] <= 1000000)]
            price_removed = initial_rows - len(data)
            if price_removed > 0:
                self.logger.info(
                    f"Removed {price_removed} rows with extreme prices (outside 50k-1M zł range)"
                )

        # Handle room count outliers (remove houses with more than 15 rooms)
        if "rooms" in data.columns:
            rooms_initial = len(data)
            data = data[data["rooms"] <= 15]
            rooms_removed = rooms_initial - len(data)
            if rooms_removed > 0:
                self.logger.info(
                    f"Removed {rooms_removed} rows with >15 rooms (unrealistic room count)"
                )
            else:
                self.logger.info("No houses with >15 rooms found to remove")

        total_removed = initial_rows - len(data)
        if total_removed > 0:
            self.logger.info(f"Total rows removed for outliers: {total_removed}")

        return data

    def _ensure_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all remaining columns are numeric"""
        non_numeric_cols = data.select_dtypes(include=["object"]).columns.tolist()
        if non_numeric_cols:
            self.logger.warning(
                f"Removing remaining non-numeric columns: {non_numeric_cols}"
            )
            data = data.drop(columns=non_numeric_cols)

        return data

    def _log_training_columns(self, data: pd.DataFrame) -> None:
        """Log all columns used for training with examples"""
        columns_without_price = [col for col in data.columns if col != "price"]

        self.logger.info(f"=== TRAINING COLUMNS SUMMARY ===")
        self.logger.info(
            f"Total columns used for training: {len(columns_without_price)}"
        )
        self.logger.info(f"Total samples available: {len(data)}")

        # Log each column with examples
        for i, col in enumerate(columns_without_price, 1):
            # Get sample values (first 3 non-null values)
            sample_values = data[col].dropna().head(3).tolist()
            data_type = str(data[col].dtype)
            unique_count = data[col].nunique()

            self.logger.info(
                f"{i:2d}. {col:<25} | Type: {data_type:<10} | Unique: {unique_count:<5} | Examples: {sample_values}"
            )

        # Log one-hot encoded columns specifically
        onehot_columns = [
            col
            for col in columns_without_price
            if any(
                original in col
                for original in [
                    "city",
                    "property_type",
                    "ownership_type",
                    "voivodeship",
                    "rooms",
                ]
            )
        ]
        if onehot_columns:
            self.logger.info(f"=== ONE-HOT ENCODED COLUMNS ({len(onehot_columns)}) ===")
            for col in onehot_columns:
                sample_values = data[col].dropna().head(3).tolist()
                self.logger.info(f"  {col}: {sample_values}")

        self.logger.info(f"=== END COLUMNS SUMMARY ===")

    def split_features_target(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Split data into features and target
        """
        if "price" in data.columns:
            X = data.drop(columns=["price"])
            y = data["price"]
            self.feature_names = X.columns.tolist()
            self.logger.info(f"Split data: {X.shape[1]} features, {X.shape[0]} samples")
            return X, y
        else:
            # For data without price column (e.g., prediction data)
            self.feature_names = data.columns.tolist()
            self.logger.info(
                "Feature data: %d features, %d samples", data.shape[1], data.shape[0]
            )
            return data, None

    def create_data_splits(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create train/validation/test splits with proper shuffling"""
        # Get split ratios
        train_ratio = self.config.get("data.loading.train_ratio", 0.7)
        val_ratio = self.config.get("data.loading.val_ratio", 0.15)
        test_ratio = self.config.get("data.loading.test_ratio", 0.15)

        shuffle_data = self.config.get("data.loading.shuffle_data", True)
        random_state = self.config.get("data.loading.random_state", 42)

        self.logger.info(
            f"Creating data splits: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

        # First split: separate test set
        test_size = test_ratio
        train_val_size = train_ratio + val_ratio

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle_data
        )

        # Second split: separate train and validation
        val_size_adjusted = val_ratio / train_val_size

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            shuffle=shuffle_data,
        )

        splits = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

        # Log split sizes
        for split_name, split_data in splits.items():
            self.logger.info(f"{split_name}: {len(split_data)} samples")

        return splits

    def normalize_features(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize features using configured scaler"""
        if not self.config.get("data.preprocessing.normalize_features", True):
            return data_splits

        scaler_type = self.config.get("data.preprocessing.scaler_type", "standard")

        # Initialize scaler
        if scaler_type == "standard":
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
        elif scaler_type == "robust":
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()
        elif scaler_type == "minmax":
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
        else:
            self.logger.warning(
                f"Unknown scaler type: {scaler_type}. Using StandardScaler."
            )
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

        # Fit scalers on training data
        X_train_scaled = scaler_X.fit_transform(data_splits["X_train"])
        y_train_scaled = scaler_y.fit_transform(
            data_splits["y_train"].values.reshape(-1, 1)
        ).ravel()

        # Transform all splits
        X_val_scaled = scaler_X.transform(data_splits["X_val"])
        X_test_scaled = scaler_X.transform(data_splits["X_test"])

        y_val_scaled = scaler_y.transform(
            data_splits["y_val"].values.reshape(-1, 1)
        ).ravel()
        y_test_scaled = scaler_y.transform(
            data_splits["y_test"].values.reshape(-1, 1)
        ).ravel()

        # Store scalers for later use
        self.scalers = {"scaler_X": scaler_X, "scaler_y": scaler_y}

        # Update data splits
        normalized_splits = {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train_scaled,
            "y_val": y_val_scaled,
            "y_test": y_test_scaled,
            "scalers": self.scalers,
        }

        self.logger.info(f"Features normalized using {scaler_type} scaler")
        return normalized_splits

    def prepare_data_pipeline(self) -> Dict[str, Any]:
        """
        Complete data preparation pipeline, now returns embedding info.
        """
        # Load data
        raw_data = self.load_data()

        # Preprocess data
        processed_data, embedding_info = self.preprocess_data(raw_data)        # Split features and target
        X, y = self.split_features_target(processed_data)
        
        # Validate that we have target data for training
        if y is None:
            raise ValueError("No target column 'price' found in data. Cannot create training splits.")

        # Create train/val/test splits
        data_splits = self.create_data_splits(X, y)

        # Normalize features
        normalized_splits = self.normalize_features(data_splits)

        # Add metadata
        normalized_splits["feature_names"] = self.feature_names
        normalized_splits["original_shape"] = raw_data.shape
        normalized_splits["processed_shape"] = processed_data.shape
        normalized_splits["embedding_info"] = embedding_info

        self.logger.info("Data pipeline completed successfully")
        return normalized_splits

    def prepare_sequence_data(self, sequence_length=4):
        """
        Przygotowuje dane w format sekwencyjny dla LSTM.
        sequence_length: liczba poprzednich lat użytych do predykcji
        """
        data = self.load_data()
        # Sortujemy dane po dacie
        data = data.sort_values('rok')

        sequences = []
        targets = []

        for location in data['lokalizacja'].unique():
            location_data = data[data['lokalizacja'] == location]
            if len(location_data) >= sequence_length + 1:
                for i in range(len(location_data) - sequence_length):
                    sequence = location_data.iloc[i:i + sequence_length]
                    target = location_data.iloc[i + sequence_length]['cena']
                    sequences.append(sequence.drop(['cena'], axis=1).values)
                    targets.append(target)

        return np.array(sequences), np.array(targets)

    def save_split_datasets_raw(self, output_base_dir: str = "data/splits") -> Dict[str, Any]:
        """
        Save training, validation, and test datasets as 3 separate CSV files with RAW data
        (before normalization but after preprocessing like categorical encoding).
        Each file contains both features and target values combined.
        Only saves if the files do not already exist.
        
        Args:
            output_base_dir: Base directory where split files will be created
            
        Returns:
            Dictionary with information about saved datasets
        """
        import os
        
        # Create base directory if it doesn't exist
        os.makedirs(output_base_dir, exist_ok=True)
        
        self.logger.info("Loading and preprocessing raw data for splitting...")
        
        # Load and preprocess data (but don't normalize)
        raw_data = self.load_data()
        processed_data, embedding_info = self.preprocess_data(raw_data)
          # Split features and target
        X, y = self.split_features_target(processed_data)
        
        # Validate that we have target data
        if y is None:
            raise ValueError("No target column 'price' found in data. Cannot create training splits.")
        
        # Create train/val/test splits (this gives us raw preprocessed data)
        raw_splits = self.create_data_splits(X, y)
        
        # Define the split names and corresponding data
        splits_info = {
            'training': {
                'X': raw_splits['X_train'],
                'y': raw_splits['y_train']
            },
            'validation': {
                'X': raw_splits['X_val'],
                'y': raw_splits['y_val']
            },
            'test': {
                'X': raw_splits['X_test'],
                'y': raw_splits['y_test']
            }
        }
        
        saved_files = []
        
        for split_name, split_data in splits_info.items():
            # Define file path - single file per split
            dataset_file = os.path.join(output_base_dir, f'{split_name}_dataset.csv')
            
            # Check if file already exists
            if os.path.exists(dataset_file):
                self.logger.info(f"Dataset file for '{split_name}' already exists. Skipping...")
                continue
            
            # Convert to DataFrames and combine
            try:
                # Handle features (X data) with original column names
                if hasattr(split_data['X'], 'columns'):
                    # If it's already a DataFrame
                    features_df = split_data['X'].copy()
                else:
                    # If it's a pandas Series or numpy array, create DataFrame with original column names
                    if self.feature_names is not None:
                        features_df = pd.DataFrame(split_data['X'], columns=self.feature_names)
                    else:
                        features_df = pd.DataFrame(split_data['X'])
                
                # Handle target (y data) - should be pandas Series
                if hasattr(split_data['y'], 'values'):
                    # If it's a pandas Series
                    target_values = split_data['y'].values
                else:
                    # If it's a numpy array
                    target_values = split_data['y']
                
                # Add target column to features DataFrame
                features_df['price'] = target_values
                
                # Save the combined dataset
                features_df.to_csv(dataset_file, index=False)
                saved_files.append(dataset_file)
                
                self.logger.info(f"Saved {split_name} dataset (RAW - before normalization):")
                self.logger.info(f"  File: {dataset_file}")
                self.logger.info(f"  Shape: {features_df.shape} (features + target)")
                self.logger.info(f"  Samples: {len(features_df)}")
                if self.feature_names:
                    self.logger.info(f"  Column names preserved: {len(self.feature_names)} features + price")
                
            except Exception as e:
                self.logger.error(f"Error saving {split_name} dataset: {str(e)}")
                
        self.logger.info(f"RAW dataset splitting completed. {len(saved_files)} files saved in: {output_base_dir}")
        self.logger.info("These datasets contain the same splits that will be used for training, but with raw values (before normalization)")
        
        # Return summary information
        return {
            "original_shape": raw_data.shape,
            "processed_shape": processed_data.shape,
            "n_features": X.shape[1],
            "n_train": len(raw_splits["X_train"]),
            "n_val": len(raw_splits["X_val"]),
            "n_test": len(raw_splits["X_test"]),
            "random_state": self.config.get("data.loading.random_state", 42),
            "train_ratio": self.config.get("data.loading.train_ratio", 0.8),
            "val_ratio": self.config.get("data.loading.val_ratio", 0.15),
            "test_ratio": self.config.get("data.loading.test_ratio", 0.05),
            "datasets_saved_to": output_base_dir,
            "saved_files": saved_files
        }