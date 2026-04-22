import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os

def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_data(data_path):
    """Load the dataset from CSV."""
    return pd.read_csv(data_path)

def introduce_missing_values(df, column='Age', percentage=0.1):
    """Introduce missing values for demonstration purposes."""
    df_copy = df.copy()  # Work on a copy to avoid modifying original
    num_missing = int(percentage * len(df_copy))
    missing_indices = np.random.choice(df_copy.index, num_missing, replace=False)
    df_copy.loc[missing_indices, column] = np.nan
    return df_copy

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline:
    - Encode target variable
    - Split into train/test sets
    - Handle missing values
    - One-hot encode categorical variables
    - Align train/test columns
    """
    # Work on a copy to avoid modifying original dataframe
    df_copy = df.copy()

    # Encode target variable
    df_copy['Attrition'] = df_copy['Attrition'].map({'Yes': 1, 'No': 0}).astype(int)

    # Split features and target
    features = df_copy.drop('Attrition', axis=1)
    target = df_copy['Attrition']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    # Handle missing values (fill with mean from train)
    mean_age = X_train['Age'].mean()
    X_train['Age'] = X_train['Age'].fillna(mean_age)
    X_test['Age'] = X_test['Age'].fillna(mean_age)

    # Encode categoricals
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Align columns
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    config = load_config()
    df = load_data(config['data']['raw_path'])
    df = introduce_missing_values(df, percentage=config['data']['missing_percentage'])
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    print("Preprocessing complete.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")