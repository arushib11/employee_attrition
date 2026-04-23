#!/usr/bin/env python3
"""
Compare Experiments Script

This script queries MLflow experiments and identifies the best performing model
based on the primary metric (F1-score for this classification task).
"""

import pandas as pd
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def compare_experiments():
    """Compare all experiments and find the best one."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # Set experiment
        mlflow.set_experiment("Employee Attrition")

        # Get experiment ID
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Employee Attrition")

        if experiment is None:
            print("❌ No 'Employee Attrition' experiment found in MLflow")
            return None

        # Search for all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            print("❌ No runs found in the experiment")
            return None

        print(f"📊 Found {len(runs)} experiment runs")
        print("\n📈 Experiment Results (sorted by F1-score):")

        # Display results sorted by F1-score (primary metric)
        results = runs[['run_id', 'metrics.f1', 'metrics.accuracy', 'metrics.precision', 'metrics.recall',
                       'params.n_estimators', 'params.max_depth', 'params.random_state']].copy()

        # Rename columns for better display
        results.columns = ['Run ID', 'F1-Score', 'Accuracy', 'Precision', 'Recall',
                          'n_estimators', 'max_depth', 'random_state']

        # Sort by F1-score descending
        results = results.sort_values('F1-Score', ascending=False)

        print(results.to_string(index=False, float_format='%.3f'))

        # Find best run
        best_run = results.iloc[0]
        print(f"\n🏆 Best Model (Run ID: {best_run['Run ID']}):")
        print(f"   F1-Score: {best_run['F1-Score']:.3f}")
        print(f"   Accuracy: {best_run['Accuracy']:.3f}")
        print(f"   Precision: {best_run['Precision']:.3f}")
        print(f"   Recall: {best_run['Recall']:.3f}")
        print(f"   Hyperparameters: n_estimators={best_run['n_estimators']}, max_depth={best_run['max_depth']}")

        return best_run

    except ImportError:
        print("⚠️  MLflow not available - cannot compare experiments")
        print("💡 To use this feature, install MLflow: pip install mlflow")
        return None
    except Exception as e:
        print(f"❌ Error comparing experiments: {e}")
        return None

def analyze_hyperparameter_impact():
    """Analyze how different hyperparameters affect performance."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        import matplotlib.pyplot as plt

        # Get experiment data
        mlflow.set_experiment("Employee Attrition")
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Employee Attrition")

        if experiment is None:
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty or len(runs) < 2:
            print("📊 Need at least 2 experiments to analyze hyperparameter impact")
            return

        print("\n📊 Hyperparameter Impact Analysis:")

        # Analyze n_estimators impact
        if 'params.n_estimators' in runs.columns and 'metrics.f1' in runs.columns:
            est_f1 = runs[['params.n_estimators', 'metrics.f1']].dropna()
            if len(est_f1) > 1:
                est_f1['params.n_estimators'] = est_f1['params.n_estimators'].astype(int)
                avg_by_est = est_f1.groupby('params.n_estimators')['metrics.f1'].mean()
                print("\nAverage F1-score by n_estimators:")
                print(avg_by_est)

        # Analyze max_depth impact
        if 'params.max_depth' in runs.columns and 'metrics.f1' in runs.columns:
            depth_f1 = runs[['params.max_depth', 'metrics.f1']].dropna()
            if len(depth_f1) > 1:
                depth_f1['params.max_depth'] = depth_f1['params.max_depth'].astype(int)
                avg_by_depth = depth_f1.groupby('params.max_depth')['metrics.f1'].mean()
                print("\nAverage F1-score by max_depth:")
                print(avg_by_depth)

    except ImportError:
        pass  # MLflow not available
    except Exception as e:
        print(f"⚠️  Could not analyze hyperparameter impact: {e}")

if __name__ == "__main__":
    print("🔍 Comparing Employee Attrition Experiments")
    print("=" * 50)

    best_run = compare_experiments()
    analyze_hyperparameter_impact()

    if best_run is not None:
        print("\n✅ Experiment comparison complete!")
        print(f"💡 Best model has F1-score: {best_run['F1-Score']:.3f}")
    else:
        print("\n💡 Run some experiments first with 'python src/train.py'")