import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Perform Exploratory Data Analysis (EDA) on the given DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    None: Displays plots and prints summaries.
    """
    # Overview of the data
    print("\n--- Dataset Overview ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\n--- Data Types ---")
    print(df.dtypes)

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(df.describe(include='all'))

    # Missing values
    print("\n--- Missing Values ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # Distribution of numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    print("\n--- Distribution of Numerical Features ---")
    df[numerical_features].hist(bins=30, figsize=(15, 10))
    plt.suptitle('Histograms of Numerical Features')
    plt.show()

    # Distribution of categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    print("\n--- Distribution of Categorical Features ---")
    for feature in categorical_features:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=feature, order=df[feature].value_counts().index)
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
        plt.show()

    # Correlation analysis
    print("\n--- Correlation Analysis ---")
    if len(numerical_features) > 1:
        corr_matrix = df[numerical_features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        plt.title('Correlation Matrix')
        plt.show()
    else:
        print("Not enough numerical features for correlation analysis.")

    # Outlier detection using box plots
    print("\n--- Outlier Detection ---")
    for feature in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=feature)
        plt.title(f'Boxplot of {feature}')
        plt.show()

    print("EDA completed successfully!")
