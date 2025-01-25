import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def calculate_woe_iv(data, feature, target):
    """
    Calculate Weight of Evidence (WOE) and Information Value (IV) for a given feature.

    Parameters:
    data (pd.DataFrame): Input DataFrame
    feature (str): Feature column name
    target (str): Target column name

    Returns:
    pd.DataFrame: DataFrame with WOE values for the feature
    """
    eps = 1e-7  # Small value to avoid division by zero
    grouped = data.groupby(feature)[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = grouped['sum'] / grouped['sum'].sum()
    grouped['non_event_rate'] = grouped['non_event'] / grouped['non_event'].sum()
    grouped['woe'] = np.log((grouped['event_rate'] + eps) / (grouped['non_event_rate'] + eps))
    grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
    return grouped[['woe', 'iv']]

def feature_engineering(df):
    """
    Perform feature engineering on the given DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    pd.DataFrame: DataFrame with engineered features.
    """

    # Create Aggregate Features
    print("\n--- Creating Aggregate Features ---")
    df['TotalTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    df['AvgTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('mean')
    df['TransactionCount'] = df.groupby('CustomerId')['Amount'].transform('count')
    df['StdTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('std')

    # Extract Date-Based Features
    print("\n--- Extracting Date-Based Features ---")
    df['TransactionHour'] = pd.to_datetime(df['TransactionStartTime']).dt.hour
    df['TransactionDay'] = pd.to_datetime(df['TransactionStartTime']).dt.day
    df['TransactionMonth'] = pd.to_datetime(df['TransactionStartTime']).dt.month
    df['TransactionYear'] = pd.to_datetime(df['TransactionStartTime']).dt.year

    # Encode Categorical Variables
    print("\n--- Encoding Categorical Variables ---")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 50]
    low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= 50]

    # One-Hot Encoding for low cardinality columns
    if low_cardinality_cols:
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        one_hot_encoded = pd.DataFrame(one_hot_encoder.fit_transform(df[low_cardinality_cols]),
                                       columns=one_hot_encoder.get_feature_names_out(low_cardinality_cols))
        df = pd.concat([df, one_hot_encoded], axis=1)

    # Label Encoding for high cardinality columns
    for col in high_cardinality_cols:
        df[f'{col}_Encoded'] = LabelEncoder().fit_transform(df[col])

    # Handle Missing Values
    print("\n--- Handling Missing Values ---")
    imputer = SimpleImputer(strategy='mean')  # Example with mean, change strategy as needed
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Normalize/Standardize Numerical Features
    print("\n--- Normalizing/Standardizing Numerical Features ---")
    scaler = MinMaxScaler()  # Use StandardScaler() for standardization
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Feature Engineering with Weight of Evidence (WOE)
    print("\n--- Feature Engineering with WOE ---")
    if 'FraudResult' in df.columns:
        for feature in categorical_cols:
            woe_iv = calculate_woe_iv(df, feature, 'FraudResult')
            df = df.merge(woe_iv['woe'], how='left', left_on=feature, right_index=True, suffixes=('', f'_WOE_{feature}'))

    print("Feature Engineering completed successfully!")
    df.head(10)
    return df
