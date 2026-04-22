import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_data, introduce_missing_values, preprocess_data

def test_load_data():
    # Test with a temporary CSV file
    test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    test_path = 'test_data.csv'
    test_df.to_csv(test_path, index=False)
    try:
        df = load_data(test_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['A', 'B']
    finally:
        os.remove(test_path)

def test_introduce_missing_values():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    df_modified = introduce_missing_values(df, 'A', 0.2)
    assert df_modified['A'].isnull().sum() > 0
    # Check that original dataframe is not modified
    assert df['A'].isnull().sum() == 0

def test_preprocess_data():
    # Mock data
    df = pd.DataFrame({
        'Age': [25, 30, 35],
        'Attrition': ['Yes', 'No', 'Yes'],
        'Department': ['Sales', 'HR', 'IT']
    })
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert len(X_train) > 0
    assert len(y_train) > 0
    assert y_train.dtype == int
    assert 0 in y_train.values and 1 in y_train.values

def test_preprocess_data_immutability():
    """Test that preprocess_data doesn't modify the original dataframe."""
    original_df = pd.DataFrame({
        'Age': [25, 30, 35],
        'Attrition': ['Yes', 'No', 'Yes'],
        'Department': ['Sales', 'HR', 'IT']
    })
    original_copy = original_df.copy()

    preprocess_data(original_df)

    # Original dataframe should be unchanged
    pd.testing.assert_frame_equal(original_df, original_copy)

def test_preprocess_data_error_handling():
    """Test error handling for invalid input."""
    # Test with missing target column
    df_no_target = pd.DataFrame({
        'Age': [25, 30, 35],
        'Department': ['Sales', 'HR', 'IT']
    })

    with pytest.raises(KeyError):
        preprocess_data(df_no_target)

def test_preprocess_data_column_alignment():
    """Test that train and test sets have aligned columns."""
    df = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45, 50],
        'Attrition': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR', 'IT']
    })

    X_train, X_test, y_train, y_test = preprocess_data(df, test_size=0.5, random_state=42)

    # Check that both sets have the same columns
    assert set(X_train.columns) == set(X_test.columns)

    # Check that columns are in the same order
    assert list(X_train.columns) == list(X_test.columns)

def test_preprocess_data_missing_value_handling():
    """Test that missing values are properly handled."""
    df = pd.DataFrame({
        'Age': [25, np.nan, 35, 40],
        'Attrition': ['Yes', 'No', 'Yes', 'No'],
        'Department': ['Sales', 'HR', 'IT', 'Sales']
    })

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Check that no missing values remain in processed data
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()

def test_preprocess_data_categorical_encoding():
    """Test that categorical variables are properly encoded."""
    df = pd.DataFrame({
        'Age': [25, 30, 35],
        'Attrition': ['Yes', 'No', 'Yes'],
        'Department': ['Sales', 'HR', 'IT']
    })

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Check that original categorical column is gone
    assert 'Department' not in X_train.columns

    # Check that dummy columns were created
    department_cols = [col for col in X_train.columns if col.startswith('Department_')]
    assert len(department_cols) > 0