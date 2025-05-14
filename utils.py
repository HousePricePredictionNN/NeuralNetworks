import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization 
from tensorflow.keras.optimizers import Adam

def visualize_missing_data(df):
    
    # Matrix plot with improved label visibility
    msno.matrix(
        df,
        figsize=(12, 6),      # Control figure dimensions
        fontsize=8,            # Larger font size for visibility
        color=(0.27, 0.52, 0.71),  # Custom bar color
        sparkline=True         # Show the sparkline at bottom
    )

    # Ensure labels are visible with more space
    plt.xticks(rotation=90, fontsize=10, ha='center')  # Increase tick font size and center
    plt.subplots_adjust(bottom=0.4)  # More space for rotated labels
    
    # Add margin to ensure labels aren't cut off
    plt.margins(x=0.01)  # Small horizontal margin
    
    plt.title("Missing Value Matrix")
    
    # Remove tight_layout as it can sometimes cut off labels
    # plt.tight_layout()
    
    plt.show()
    
    # As backup, print column names to console
    print("\nColumns in dataset:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}: {col}")
        

def analyze_missing_data(df):
    missing_percentage = df.isnull().mean() * 100

    missing_percentage.sort_values(ascending=False).plot(kind='bar')
    plt.title('Percentage of Missing Values by Column')
    plt.ylabel('Percentage (%)')
    plt.axhline(y=5, color='r', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return missing_percentage     

def add_missing_indicators(df, threshold=0.5):
    missing_percentage = df.isnull().mean() * 100
    columns_with_missing_data = missing_percentage[missing_percentage > threshold].index.tolist()

    df_with_inidicators = df.copy()

    for col in columns_with_missing_data:
        df_with_inidicators[f"{col}_missing"] = df_with_inidicators[col].isnull().astype(int)

    return df_with_inidicators    

def preprocess_data(df, numeric_cols, categorical_cols, bool_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', 'passthrough'),  # Will handle missing values as a category in OHE
        ('scaler', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    bool_transofmer = Pipeline(steps=[
            ('imputer', 'passthrough')
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('bool', bool_transofmer, bool_cols)
        ])
    
    return preprocessor

def impute_location_specific_data(df):
    pass

def prepare_data_for_nn(df, test_size=0.2, random_state=42):

    df_with_inidcators = add_missing_indicators(df, threshold=5.0)

    X = df_with_inidcators.drop(['price'], axis=1)
    y = df_with_inidcators['price']

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    preprocessor = preprocess_data(X_train, numeric_cols, categorical_cols, bool_cols)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor

def build_neural_network(input_dim, learning_reate=0.001):

    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  
    ])

    optimizer = Adam(learning_rate=learning_reate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

