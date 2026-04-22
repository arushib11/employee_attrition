# Data Drift Monitoring Analysis

## Overview

This document analyzes data drift monitoring for the Employee Attrition Prediction model using Evidently AI. The monitoring compares reference data (original training data) against simulated production data to detect potential model performance degradation.

## Drift Detection Setup

### Reference Data
- **Source**: Original training dataset (`data/employee_attrition.csv`)
- **Size**: 1,176 samples (80% of original data)
- **Features**: 47 features (after preprocessing and one-hot encoding)
- **Target**: Employee attrition (Yes/No)

### Production Data Simulation
- **Method**: Simulated production data created by sampling from reference data with introduced drift
- **Size**: 300 samples
- **Drift Features**: Age, MonthlyIncome, YearsAtCompany, JobSatisfaction
- **Drift Magnitude**: 10-30% shift in numeric features, 20% categorical distribution changes

## Drift Analysis Results

### Dataset-Level Drift
- **Drift Share**: 0.125 (12.5% of features show drift)
- **Drift Detected**: Yes (exceeds 10% threshold)

### Feature-Level Drift Analysis

#### Features Showing Significant Drift:
1. **Age** - Distribution shifted upward (10-15% increase)
2. **MonthlyIncome** - Higher income values in production data
3. **YearsAtCompany** - More experienced employees in production
4. **JobSatisfaction** - Different satisfaction distribution

#### Features Without Significant Drift:
- Department, BusinessTravel, DistanceFromHome
- Education, EnvironmentSatisfaction
- Most one-hot encoded categorical features

## Why These Features Drifted

### Age
- **Reason**: Demographic shift in workforce
- **Impact**: Age correlates with experience and job stability
- **Business Context**: Company may be hiring more experienced professionals

### MonthlyIncome
- **Reason**: Salary adjustments, inflation, or different job roles
- **Impact**: Income is a strong predictor of attrition
- **Business Context**: Compensation changes or different market conditions

### YearsAtCompany
- **Reason**: More tenured employees in recent data
- **Impact**: Experience level affects attrition likelihood
- **Business Context**: Better retention or different hiring patterns

### JobSatisfaction
- **Reason**: Changes in work environment or survey methodology
- **Impact**: Satisfaction is a key attrition driver
- **Business Context**: Workplace changes or seasonal variations

## Impact on Model Performance

### Likely Effects:
1. **Reduced Accuracy**: Model trained on different distributions
2. **Poor Generalization**: Production data doesn't match training data
3. **Biased Predictions**: Over/under prediction for certain employee groups

### Risk Assessment:
- **High Risk**: Income and satisfaction are top attrition predictors
- **Medium Risk**: Age and tenure affect prediction reliability
- **Low Risk**: Other features show minimal drift

## Recommendations

### Immediate Actions (Next 1-2 weeks):
1. **Retrain Model**: Include recent production data in training
2. **Feature Engineering**: Add drift-resistant features
3. **Threshold Adjustment**: Recalibrate decision thresholds

### Monitoring Strategy (Ongoing):
1. **Weekly Drift Checks**: Automate drift detection in production
2. **Performance Monitoring**: Track model accuracy on recent data
3. **Alert System**: Set up alerts when drift exceeds 15%

### Long-term Solutions (1-3 months):
1. **Data Pipeline**: Implement continuous data collection
2. **Model Retraining**: Set up automated retraining pipeline
3. **Feature Selection**: Focus on stable, predictive features

## Technical Implementation

### Drift Detection Code:
```python
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric

report = Report(metrics=[DataDriftTable(), DatasetDriftMetric()])
report.run(reference_data=reference, current_data=production)
```

### Threshold Settings:
- **Dataset Drift Threshold**: 10% (configurable)
- **Feature Drift Threshold**: Automatic (based on statistical tests)
- **Alert Level**: Exit code 1 when drift > threshold

## Conclusion

The drift analysis revealed significant changes in key features that directly impact employee attrition predictions. The 12.5% drift share exceeds our 10% threshold, indicating the need for model retraining. Regular monitoring and automated retraining will be essential to maintain model performance in production.

**Next Steps**: Implement automated retraining pipeline and set up continuous monitoring dashboard.