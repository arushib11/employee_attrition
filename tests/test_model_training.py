import pytest
import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation import evaluate_model
from model_training import train_model

def test_train_model():
    # Mock data
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    model = train_model(X_train, y_train, n_estimators=10, max_depth=5, random_state=42)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    # Mock model and data
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    model.fit(X_train, y_train)
    metrics, report = evaluate_model(model, X_train, y_train)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert all(v >= 0.0 for v in metrics.values())
    assert isinstance(report, str)