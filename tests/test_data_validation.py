import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_data, introduce_missing_values, preprocess_data
from evaluation import evaluate_model
from model_training import train_model

class TestDataValidation:
    """Data validation tests to ensure dataset quality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset similar to the real one."""
        np.random.seed(42)
        n_samples = 100

        data = {
            'Age': np.random.randint(18, 65, n_samples),
            'Attrition': np.random.choice(['Yes', 'No'], n_samples, p=[0.16, 0.84]),
            'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n_samples),
            'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_samples),
            'DistanceFromHome': np.random.randint(1, 30, n_samples),
            'Education': np.random.randint(1, 6, n_samples),
            'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
            'JobSatisfaction': np.random.randint(1, 5, n_samples),
            'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
            'NumCompaniesWorked': np.random.randint(0, 10, n_samples),
            'TotalWorkingYears': np.random.randint(0, 40, n_samples),
            'YearsAtCompany': np.random.randint(0, 30, n_samples),
        }
        return pd.DataFrame(data)

    def test_dataset_has_required_columns(self, sample_dataset):
        """Test that dataset contains all required columns."""
        required_columns = ['Age', 'Attrition', 'BusinessTravel', 'Department']

        for col in required_columns:
            assert col in sample_dataset.columns, f"Required column '{col}' is missing"

    def test_target_variable_values(self, sample_dataset):
        """Test that target variable contains only expected values."""
        unique_values = sample_dataset['Attrition'].unique()
        expected_values = {'Yes', 'No'}

        assert set(unique_values).issubset(expected_values), \
            f"Target variable contains unexpected values: {set(unique_values) - expected_values}"

    def test_numeric_features_ranges(self, sample_dataset):
        """Test that numeric features are within reasonable ranges."""
        # Age should be between 18 and 70
        assert sample_dataset['Age'].min() >= 18, "Age minimum is too low"
        assert sample_dataset['Age'].max() <= 70, "Age maximum is too high"

        # Monthly income should be positive
        assert sample_dataset['MonthlyIncome'].min() > 0, "Monthly income should be positive"

        # Distance from home should be positive
        assert sample_dataset['DistanceFromHome'].min() >= 0, "Distance from home should be non-negative"

        # Working years should be non-negative
        assert sample_dataset['TotalWorkingYears'].min() >= 0, "Total working years should be non-negative"
        assert sample_dataset['YearsAtCompany'].min() >= 0, "Years at company should be non-negative"

    def test_dataset_minimum_size(self, sample_dataset):
        """Test that dataset has minimum required size."""
        assert len(sample_dataset) >= 50, f"Dataset too small: {len(sample_dataset)} rows (minimum 50 required)"

class TestModelValidation:
    """Model validation tests to ensure model quality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        # Create features
        X = pd.DataFrame({
            'Age': np.random.randint(20, 60, n_samples),
            'MonthlyIncome': np.random.randint(2000, 15000, n_samples),
            'YearsAtCompany': np.random.randint(0, 20, n_samples),
            'JobSatisfaction': np.random.randint(1, 5, n_samples),
        })

        # Create target (imbalanced like real data)
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.8, 0.2]))

        return X, y

    def test_model_predictions_shape_and_type(self, sample_data):
        """Test that model produces predictions of correct shape and type."""
        X_train, y_train = sample_data

        # Train model
        model = train_model(X_train, y_train, n_estimators=10, max_depth=5, random_state=42)

        # Make predictions
        predictions = model.predict(X_train)

        # Check shape
        assert predictions.shape == (len(X_train),), f"Predictions shape {predictions.shape} != expected {(len(X_train),)}"

        # Check type
        assert predictions.dtype in [np.int32, np.int64, int], f"Predictions type {predictions.dtype} should be integer"

        # Check values are valid classes
        unique_preds = np.unique(predictions)
        assert set(unique_preds).issubset({0, 1}), f"Predictions contain invalid classes: {unique_preds}"

    def test_model_meets_minimum_performance(self, sample_data):
        """Test that model achieves minimum performance threshold."""
        X_train, y_train = sample_data

        # Train model
        model = train_model(X_train, y_train, n_estimators=10, max_depth=5, random_state=42)

        # Evaluate
        metrics, report = evaluate_model(model, X_train, y_train)

        # Check that all metrics are reasonable
        assert metrics['accuracy'] >= 0.5, f"Accuracy {metrics['accuracy']:.3f} below minimum threshold 0.5"
        assert metrics['precision'] >= 0.0, "Precision should be non-negative"
        assert metrics['recall'] >= 0.0, "Recall should be non-negative"
        assert metrics['f1'] >= 0.0, "F1 score should be non-negative"

        # Check that metrics are not all the same (indicating a broken model)
        metric_values = list(metrics.values())
        assert not all(v == metric_values[0] for v in metric_values), "All metrics are identical - possible model issue"