#!/usr/bin/env python3
"""
Main training script for Employee Attrition Prediction.

This script orchestrates the complete MLOps pipeline:
1. Load and preprocess data
2. Train model
3. Evaluate and log experiment
4. Validate against thresholds
"""

import sys
import os

# Ensure src directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np
from data_preprocessing import load_config, load_data, introduce_missing_values, preprocess_data
from model_training import train_model, evaluate_model
from utils import get_project_root, get_dvc_data_md5

try:
    from model_training import log_experiment
    MLFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  MLflow not available - experiment logging disabled")
    MLFLOW_AVAILABLE = False

def validate_model_performance(metrics, thresholds):
    """Validate model meets minimum performance thresholds."""
    if metrics['accuracy'] < thresholds['min_accuracy']:
        print(f"❌ Model accuracy {metrics['accuracy']:.3f} below threshold {thresholds['min_accuracy']}")
        return False

    if metrics['f1'] < thresholds['min_f1']:
        print(f"❌ Model F1-score {metrics['f1']:.3f} below threshold {thresholds['min_f1']}")
        return False

    print("✅ Model meets all performance thresholds")
    return True

def main():
    """Main training pipeline."""
    print("🚀 Starting Employee Attrition Training Pipeline")

    # Load configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "config.yaml")
    config = load_config(config_path)
    print("✅ Configuration loaded")

    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    rs = config["training"]["random_state"]
    np.random.seed(rs)
    df = load_data(config['data']['raw_path'])
    df = introduce_missing_values(df, percentage=config['data']['missing_percentage'])
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    print(f"✅ Data preprocessed: Train shape {X_train.shape}, Test shape {X_test.shape}")

    # Train model
    print("🤖 Training model...")
    model = train_model(
        X_train, y_train,
        n_estimators=config['model']['hyperparameters']['n_estimators'],
        max_depth=config['model']['hyperparameters']['max_depth'],
        random_state=config['model']['hyperparameters']['random_state']
    )
    print("✅ Model trained")

    # Evaluate model
    print("📈 Evaluating model...")
    metrics, report = evaluate_model(model, X_test, y_test)
    print("📊 Model Performance:")
    for metric, value in metrics.items():
        print(f"- {metric.capitalize()}: {value:.3f}")
    print("\nDetailed Classification Report:")
    print(report)

    # Validate against thresholds
    if not validate_model_performance(metrics, config['thresholds']):
        print("❌ Model validation failed - exiting with error")
        sys.exit(1)

    # Log experiment
    if MLFLOW_AVAILABLE:
        print("📝 Logging experiment to MLflow...")
        try:
            project_root = get_project_root()
            data_dvc_path = os.path.join(project_root, "data", "employee_attrition.csv.dvc")
            data_version = get_dvc_data_md5(data_dvc_path) or "unknown"
            log_experiment(model, metrics, config, data_version=data_version)
            print("✅ Experiment logged successfully")
        except Exception as exc:
            print(f"⚠️  MLflow logging failed (training still succeeded): {exc}")
    else:
        print("📝 Skipping MLflow logging (not available)")

    print("🎉 Training pipeline completed successfully!")

if __name__ == "__main__":
    main()