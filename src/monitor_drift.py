#!/usr/bin/env python3
"""
Drift Monitoring Script

This script monitors for data drift between reference (training) data
and production data using Evidently AI.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_data, introduce_missing_values, preprocess_data

def create_simulated_production_data(reference_data, n_samples=500, drift_features=None):
    """
    Create simulated production data with potential drift.

    Args:
        reference_data: Original training data
        n_samples: Number of production samples to generate
        drift_features: List of features to introduce drift in

    Returns:
        DataFrame with simulated production data
    """
    np.random.seed(42)

    # Sample from reference data
    production_data = reference_data.sample(n=n_samples, replace=True).copy()

    # Introduce drift in specified features
    if drift_features:
        for feature in drift_features:
            if feature in production_data.columns:
                if production_data[feature].dtype in ['int64', 'float64']:
                    # Add drift by shifting numeric features
                    shift = np.random.uniform(0.1, 0.3)  # 10-30% shift
                    production_data[feature] = production_data[feature] * (1 + shift)
                elif production_data[feature].dtype == 'object':
                    # For categorical features, change distribution slightly
                    unique_vals = production_data[feature].unique()
                    if len(unique_vals) > 1:
                        # Swap some values to simulate drift
                        mask = np.random.random(len(production_data)) < 0.2  # 20% of data
                        swap_indices = np.random.choice(len(unique_vals), size=mask.sum())
                        production_data.loc[mask, feature] = unique_vals[swap_indices]

    return production_data

def monitor_drift(reference_data, production_data, threshold=0.1):
    """
    Monitor for data drift using Evidently.

    Args:
        reference_data: Reference dataset (training data)
        production_data: Production dataset to compare
        threshold: Drift threshold (share of drifted features)

    Returns:
        dict: Drift analysis results
    """
    try:
        from evidently.report import Report
        from evidently.metrics import DataDriftTable
        from evidently.metrics import DatasetDriftMetric

        # Create drift report
        report = Report(metrics=[
            DataDriftTable(),
            DatasetDriftMetric()
        ])

        # Run the report
        report.run(reference_data=reference_data, current_data=production_data)

        # Get drift metrics
        drift_results = {}

        # Extract dataset drift
        dataset_drift = report.as_dict()['metrics'][1]['result']
        drift_results['dataset_drift_share'] = dataset_drift['drift_share']
        drift_results['dataset_drift_detected'] = dataset_drift['drift_detected']

        # Extract feature-level drift
        feature_drift = report.as_dict()['metrics'][0]['result']
        drift_results['feature_drift'] = {}

        for feature, drift_info in feature_drift['drift_by_columns'].items():
            drift_results['feature_drift'][feature] = {
                'drift_detected': drift_info['drift_detected'],
                'drift_score': drift_info.get('drift_score', None)
            }

        # Count drifted features
        drifted_features = [f for f, info in drift_results['feature_drift'].items() if info['drift_detected']]
        drift_results['drifted_features_count'] = len(drifted_features)
        drift_results['drifted_features'] = drifted_features

        # Save HTML report
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)

        report_path = reports_dir / 'drift_report.html'
        report.save_html(str(report_path))
        drift_results['report_path'] = str(report_path)

        return drift_results

    except ImportError:
        print("⚠️  Evidently not available - drift monitoring disabled")
        print("💡 Install Evidently: pip install evidently")
        return {
            'error': 'Evidently not available',
            'dataset_drift_share': 0.0,
            'drifted_features_count': 0
        }

def analyze_drift_results(drift_results, threshold=0.1):
    """
    Analyze drift results and provide recommendations.

    Args:
        drift_results: Results from drift monitoring
        threshold: Acceptable drift threshold
    """
    print("\n📊 Drift Analysis Results")
    print("=" * 50)

    if 'error' in drift_results:
        print("❌ Could not perform drift analysis")
        return False

    drift_share = drift_results['dataset_drift_share']
    drifted_count = drift_results['drifted_features_count']
    drifted_features = drift_results['drifted_features']

    print(".3f")
    print(f"📈 Drifted Features: {drifted_count}")

    if drifted_features:
        print(f"🎯 Features with Drift: {', '.join(drifted_features[:5])}")
        if len(drifted_features) > 5:
            print(f"   ... and {len(drifted_features) - 5} more")

    # Check against threshold
    if drift_share > threshold:
        print(f"❌ DRIFT DETECTED: Drift share {drift_share:.3f} exceeds threshold {threshold}")
        print("\n💡 Recommendations:")
        print("   • Retrain model with recent data")
        print("   • Investigate root cause of drift")
        print("   • Monitor these features closely")
        return True  # Drift detected
    else:
        print(f"✅ No significant drift detected (below {threshold} threshold)")
        print("\n💡 Recommendations:")
        print("   • Continue monitoring")
        print("   • Model performance should be acceptable")
        return False  # No significant drift

def main():
    """Main drift monitoring pipeline."""
    print("🔍 Starting Drift Monitoring")
    print("=" * 40)

    # Load reference data (training data)
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print("❌ Config file not found")
        sys.exit(1)

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    print("📊 Loading reference data...")
    reference_raw = load_data(config['data']['raw_path'])
    reference_raw = introduce_missing_values(reference_raw, percentage=config['data']['missing_percentage'])

    # Preprocess reference data
    X_ref, _, y_ref, _ = preprocess_data(
        reference_raw,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    # Combine features and target for drift analysis
    reference_data = X_ref.copy()
    reference_data['Attrition'] = y_ref

    print(f"✅ Reference data loaded: {len(reference_data)} samples")

    # Create simulated production data with drift
    print("🏭 Generating simulated production data...")
    drift_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction']
    production_raw = create_simulated_production_data(reference_raw, n_samples=300, drift_features=drift_features)

    # Preprocess production data
    X_prod, _, y_prod, _ = preprocess_data(
        production_raw,
        test_size=0,  # Use all data for production
        random_state=config['training']['random_state']
    )

    # Combine features and target
    production_data = X_prod.copy()
    production_data['Attrition'] = y_prod

    print(f"✅ Production data generated: {len(production_data)} samples")

    # Monitor for drift
    print("🔍 Analyzing data drift...")
    drift_results = monitor_drift(reference_data, production_data, threshold=0.1)

    # Analyze results
    drift_detected = analyze_drift_results(drift_results)

    # Exit with appropriate code
    if drift_detected:
        print("\n🚨 ALERT: Significant drift detected!")
        sys.exit(1)  # Non-zero exit for CI/CD
    else:
        print("\n✅ No significant drift detected")
        sys.exit(0)

if __name__ == "__main__":
    main()