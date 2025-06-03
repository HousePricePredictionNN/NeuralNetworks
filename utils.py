import os
import datetime
import pandas as pd
import numpy as np

def create_output_directory():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "output", current_date)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path    

def drop_columns_from_config(data, config):
    """Drop columns specified in config"""
    columns_to_drop = config.get('data.columns_to_drop', []) if config else []
    existing_cols_to_drop = [col for col in columns_to_drop if col in data.columns]
    
    if existing_cols_to_drop:
        data = data.drop(columns=existing_cols_to_drop)
        print(f"Dropped {len(existing_cols_to_drop)} columns from config:")
        for col in existing_cols_to_drop[:5]:  # Show first 5
            print(f"  - {col}")
        if len(existing_cols_to_drop) > 5:
            print(f"  ... and {len(existing_cols_to_drop) - 5} more")
    
    return data

def convert_comma_decimals(data):
    """Convert comma-separated decimal numbers to proper float format"""
    for col in data.select_dtypes(include=['object']).columns:
        try:
            if data[col].astype(str).str.contains(',').any():
                print(f"Converting comma decimals in column: {col}")
                data[col] = data[col].astype(str).str.replace(',', '.').replace('nan', np.nan)
                data[col] = pd.to_numeric(data[col], errors='coerce')
        except:
            pass
    return data

def convert_boolean_columns(data):
    """Convert boolean columns to integers"""
    bool_cols = data.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        data[col] = data[col].astype(int)
        print(f"Converted boolean column to int: {col}")
    return data

def convert_yes_no_columns(data):
    """Convert yes/no string columns to numeric (1/0)"""
    for col in data.select_dtypes(include=['object']).columns:
        if data[col].dropna().isin(['yes', 'no', 'Yes', 'No', 'YES', 'NO']).all():
            print(f"Converting yes/no column to numeric: {col}")
            data[col] = data[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0})
    return data

def remove_non_numeric_columns(data):
    """Remove any remaining non-numeric columns"""
    non_numeric_cols = data.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"WARNING: Removing remaining non-numeric columns: {non_numeric_cols}")
        data = data.drop(columns=non_numeric_cols)
    return data

def fill_missing_values(data):
    """Fill missing values with median for numeric columns"""
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            if data[col].isna().all():  # If entire column is NaN
                data[col] = data[col].fillna(0)
                print(f"Filled all-NaN column '{col}' with 0")
            else:
                median_value = data[col].median()
                data[col] = data[col].fillna(median_value)
                missing_count = data[col].isna().sum()
                if missing_count > 0:
                    print(f"Filled {missing_count} missing values in '{col}' with median: {median_value}")
    return data

def split_features_target(data):
    """Split data into features (X) and target (y)"""
    if 'price' in data.columns:
        X = data.drop(columns=['price'])
        y = data['price']
        print(f"Final feature set: {X.shape[1]} features and {X.shape[0]} samples")
        print(f"Final data types: {X.dtypes.value_counts()}")
        return X, y
    else:
        # For verification data without price column
        X = data
        print(f"Final feature set: {X.shape[1]} features and {X.shape[0]} samples")
        print(f"Final data types: {X.dtypes.value_counts()}")
        return X

def preprocess_data(data, config=None):
    print("Starting preprocessing...")
    print(f"Original data shape: {data.shape}")
    
    # Step 1: Drop columns specified in config
    data = drop_columns_from_config(data, config)
    
    # Step 2: Convert comma decimals to proper format
    data = convert_comma_decimals(data)
    
    # Step 3: Convert boolean columns to integers
    data = convert_boolean_columns(data)
    
    # Step 4: Convert yes/no columns to numeric
    data = convert_yes_no_columns(data)
    
    # Step 5: Remove any remaining non-numeric columns
    data = remove_non_numeric_columns(data)
    
    # Step 6: Fill missing values
    data = fill_missing_values(data)
    
    # Step 7: Split features and target
    return split_features_target(data)

def prepare_test_data(config):
    """
    Prepare test data using exact row specification from config
    
    Args:
        config: Config object with test data settings
        
    Returns:
        verification_data_set: Features for verification
        verification_data_set_expected_prices: Target values for verification
    """
    # Get config parameters
    csv_filename = config.get_data_path()
    start_row = config.get('data.verification.start_row', 6001)
    end_row = config.get('data.verification.end_row', 6501)
    
    # Calculate number of rows to load
    num_rows = end_row - start_row
    
    print(f"Preparing test data:")
    print(f"  CSV file: {csv_filename}")
    print(f"  Row range: {start_row} to {end_row-1} ({num_rows} rows)")
    
    # Load specific rows
    verification_data = pd.read_csv(
        csv_filename, 
        sep=None,
        engine='python',  # Use python engine for flexible separator
        skiprows=range(1, start_row),  # Skip header (0) and rows 1 to start_row-1
        nrows=num_rows
    )
    
    # Check if we have data
    if len(verification_data) == 0:
        raise ValueError(f"No data loaded. Check your CSV file and row range {start_row}-{end_row}")
    
    if len(verification_data) < num_rows:
        print(f"Warning: Only {len(verification_data)} rows available (requested {num_rows})")
    
    print(verification_data.columns.tolist())
    # Split features and target
    if 'price' not in verification_data.columns:
        raise ValueError("Price column not found in verification data")
    
    verification_data_set = verification_data.drop(columns=['price'])
    verification_data_set_expected_prices = verification_data['price']
    
    print(f"Loaded {len(verification_data)} verification samples")
    
    return verification_data_set, verification_data_set_expected_prices