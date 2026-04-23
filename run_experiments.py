#!/usr/bin/env python3
"""
Run Multiple Experiments Script

This script runs multiple experiments with different hyperparameters
to demonstrate experiment tracking and comparison capabilities.
"""

import yaml
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_config, load_data, introduce_missing_values, preprocess_data
from model_training import train_model, evaluate_model
from utils import get_project_root, get_dvc_data_md5

def run_experiment(config, experiment_name, n_estimators, max_depth):
    """Run a single experiment with specified hyperparameters."""
    print(f"\n🧪 Running experiment: {experiment_name}")
    print(f"   n_estimators: {n_estimators}, max_depth: {max_depth}")

    # Load and preprocess data
    df = load_data(config['data']['raw_path'])
    df = introduce_missing_values(df, percentage=config['data']['missing_percentage'])
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    # Train model with specific hyperparameters
    model = train_model(
        X_train, y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=config['model']['hyperparameters']['random_state']
    )

    # Evaluate model
    metrics, report = evaluate_model(model, X_test, y_test)

    print(f"   F1-Score: {metrics['f1']:.3f}")
    # Log experiment (will be skipped if MLflow not available)
    try:
        from model_training import log_experiment
        project_root = get_project_root()
        data_dvc_path = os.path.join(project_root, "data", "employee_attrition.csv.dvc")
        data_version = get_dvc_data_md5(data_dvc_path) or "unknown"
        log_experiment(model, metrics, config, data_version=data_version, run_name=experiment_name)
        print("   ✅ Logged to MLflow")
    except ImportError:
        print("   ⚠️  MLflow not available - experiment not logged")

    return metrics

def main():
    """Run multiple experiments with different configurations."""
    print("🚀 Running Multiple Employee Attrition Experiments")
    print("=" * 60)

    # Load base configuration
    config = load_config()

    # Define experiment configurations
    experiments = [
        {"name": "baseline", "n_estimators": 100, "max_depth": 10},
        {"name": "deep_trees", "n_estimators": 100, "max_depth": 20},
        {"name": "shallow_trees", "n_estimators": 100, "max_depth": 5},
        {"name": "many_estimators", "n_estimators": 200, "max_depth": 10},
        {"name": "few_estimators", "n_estimators": 50, "max_depth": 10},
        {"name": "balanced_config", "n_estimators": 150, "max_depth": 15},
    ]

    results = []

    for exp in experiments:
        # Update config with experiment parameters
        exp_config = config.copy()
        exp_config['model']['hyperparameters']['n_estimators'] = exp['n_estimators']
        exp_config['model']['hyperparameters']['max_depth'] = exp['max_depth']

        # Run experiment
        metrics = run_experiment(exp_config, exp['name'], exp['n_estimators'], exp['max_depth'])

        # Store results
        results.append({
            'experiment': exp['name'],
            'n_estimators': exp['n_estimators'],
            'max_depth': exp['max_depth'],
            **metrics
        })

    # Display summary
    print("\n📊 Experiment Summary:")
    print("-" * 80)
    print(f"{'Experiment':<15} {'n_est':<6} {'max_d':<6} {'Accuracy':<9} {'Precision':<10} {'Recall':<7} {'F1':<6}")
    print("-" * 80)

    for result in results:
        print(f"{result['experiment']:<15} {result['n_estimators']:<6} {result['max_depth']:<6} {result['accuracy']:<9.3f} {result['precision']:<10.3f} {result['recall']:<7.3f} {result['f1']:<6.3f}")

    # Find best experiment
    best_exp = max(results, key=lambda x: x['f1'])
    print(f"\n🏆 Best Experiment:")
    print(f"   {best_exp['experiment']} (F1: {best_exp['f1']:.3f})")
    print(f"   n_estimators: {best_exp['n_estimators']}, max_depth: {best_exp['max_depth']}")

    print(f"\n✅ All experiments completed!")
    print("💡 Use 'python compare_experiments.py' to analyze results in MLflow UI")

if __name__ == "__main__":
    main()