import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import yaml
import os
from data_preprocessing import load_config, load_data, introduce_missing_values, preprocess_data

def train_model(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Train a RandomForest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return multiple metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    report = classification_report(y_test, y_pred)
    return metrics, report

def log_experiment(model, metrics, config, data_version="v1.0"):
    """Log experiment with MLflow."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("⚠️  MLflow not available - skipping experiment logging")
        return

    mlflow.set_experiment("Employee Attrition")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(config['model']['hyperparameters'])
        mlflow.log_param("data_version", data_version)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    config = load_config()
    df = load_data(config['data']['raw_path'])
    df = introduce_missing_values(df, percentage=config['data']['missing_percentage'])
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    model = train_model(
        X_train, y_train,
        n_estimators=config['model']['hyperparameters']['n_estimators'],
        max_depth=config['model']['hyperparameters']['max_depth'],
        random_state=config['model']['hyperparameters']['random_state']
    )

    metrics, report = evaluate_model(model, X_test, y_test)
    print(f"Model Metrics: {metrics}")
    print("Classification Report:")
    print(report)

    log_experiment(model, metrics, config)
    print("Experiment logged to MLflow.")