# Employee Attrition Prediction

This project implements an end-to-end MLOps pipeline for predicting employee attrition using machine learning. It demonstrates best practices in version control, experiment tracking, automated testing, CI/CD, and drift monitoring.

## Dataset

The dataset used is the IBM HR Analytics Employee Attrition & Performance dataset from Kaggle. It contains employee information and whether they left the company (attrition).

- **Source**: [Kaggle - IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Task**: Binary classification (predict attrition: Yes/No)
- **Features**: 34 features including demographic, job-related, and satisfaction metrics
- **Rows**: 1,470

## Project Structure

```
.
├── configs/
│   └── config.yaml              # Configuration file for hyperparameters and paths
├── data/
│   ├── employee_attrition.csv   # Raw dataset (tracked with DVC)
│   └── employee_attrition.csv.dvc  # DVC pointer file
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── model_training.py        # Model training and MLflow logging
│   ├── evaluation.py            # Shared evaluation metrics and reports
│   ├── train.py                 # Main training orchestration script
│   ├── monitor_drift.py         # Drift monitoring with Evidently
│   └── utils.py                 # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   └── test_data_validation.py   # Data and model validation tests
├── reports/                     # Drift monitoring reports
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # GitHub Actions CI/CD pipeline
├── models/                      # Trained models (ignored by git)
├── mlruns/                      # MLflow experiment runs (ignored by git)
├── compare_experiments.py       # Experiment comparison script
├── run_experiments.py           # Multiple experiments runner
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── .dvc/                       # DVC configuration
├── README.md                   # This file
├── MONITORING.md               # Drift monitoring analysis
└── dvc.yaml                    # DVC pipeline configuration
```

## Setup

### Prerequisites
- Python 3.9+ (CI uses 3.9; local dev tested with Python 3.12)
- Git
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/employee_attrition.git
   cd employee_attrition
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -U pip
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up DVC and pull data**:
   ```bash
   # Pull the dataset
   # If your DVC remote is a public S3 bucket, use unsigned requests:
   AWS_NO_SIGN_REQUEST=1 dvc pull
   ```
   Notes:
   - This project uses a **public S3-backed DVC remote**. You should not need AWS credentials to pull data.
   - DVC S3 support is included in `requirements.txt`. You should not need extra installs beyond `pip install -r requirements.txt`.

5. **Optional: MLflow UI** (for experiment tracking):
   ```bash
   # Start the UI (use python -m mlflow to ensure you use the venv's mlflow)
   python -m mlflow ui --backend-store-uri "file:$(pwd)/mlruns"
   ```

## Usage

### Training a Model

**Single training run**:
```bash
PYTHONPATH=$PWD/src python src/train.py
```

**Run multiple experiments** (recommended for comparison):
```bash
python run_experiments.py
```

**Compare experiments** (after running multiple experiments):
```bash
python compare_experiments.py
```

### Testing

**Run the full test suite**:
```bash
PYTHONPATH=$PWD/src pytest tests/ -v
```

**Run with coverage**:
```bash
pytest tests/ --cov=src --cov-report=html
```

### Drift Monitoring

**Monitor for data drift**:
```bash
python src/monitor_drift.py
```

This will:
- Compare reference (training) data vs simulated production data
- Generate an HTML report in `reports/drift_report.html`
- Exit with code 1 if drift exceeds threshold

### Configuration

Modify `configs/config.yaml` to change:
- Model hyperparameters (n_estimators, max_depth)
- File paths and data settings
- Training parameters (test_size, random_state)
- Performance thresholds

Example configuration:
```yaml
data:
  raw_path: "data/employee_attrition.csv"
  missing_percentage: 0.1

model:
  type: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42

training:
  test_size: 0.2
  random_state: 42

thresholds:
  min_accuracy: 0.75
  min_f1: 0.70
```

## MLOps Components

### 1. Version Control (Git + DVC)
- **Git**: Code versioning and collaboration
- **DVC**: Data versioning and pipeline tracking
- **Remote Storage**: Public S3 bucket (used by graders to `dvc pull`)

### 2. Experiment Tracking (MLflow)
- **Automatic Logging**: Hyperparameters, metrics, and models
- **Experiment Comparison**: Query and compare multiple runs
- **Model Registry**: Store and version trained models

### 3. Automated Testing (pytest)
- **Unit Tests**: 8 tests for preprocessing functions
- **Data Validation**: 4 tests for dataset quality
- **Model Validation**: 2 tests for model performance
- **16 total tests** covering all critical functionality

### 4. CI/CD Pipeline (GitHub Actions)
- **Test Job**: Runs on every push/PR to main
- **Training Job**: Runs after tests pass, validates model performance
- **Automated Quality Gates**: Prevents poor models from deployment

### 5. Drift Monitoring (Evidently)
- **Reference vs Production**: Compares data distributions
- **Feature-level Analysis**: Identifies which features drifted
- **HTML Reports**: Visual drift detection reports
- **Automated Alerts**: Fails pipeline when drift exceeds threshold

## Development Workflow

1. **Make Changes**: Modify code, configs, or add features
2. **Run Tests**: `pytest tests/` to ensure nothing breaks
3. **Train Model**: `python src/train.py` to validate changes
4. **Commit & Push**: GitHub Actions will run CI/CD automatically
5. **Monitor Drift**: `python src/monitor_drift.py` regularly

## Key Features

- **Modular Architecture**: Separated concerns (preprocessing, training, monitoring)
- **Configuration-Driven**: All settings in YAML files
- **Comprehensive Testing**: 16 tests covering unit, data, and model validation
- **Experiment Tracking**: Compare multiple model configurations
- **Data Versioning**: DVC tracks dataset changes
- **Drift Detection**: Automated monitoring for production data changes
- **CI/CD Ready**: Automated testing and training validation

## Performance Metrics

Current model performance (Random Forest):
- **Accuracy**: ~87%
- **Precision**: ~84%
- **Recall**: ~55%
- **F1-Score**: ~67%

*Note: Performance may vary with different configurations and random seeds.*

## Troubleshooting

### Common Issues

**DVC Pull Fails**:
```bash
# Check DVC remote configuration
dvc remote list

# Reconfigure remote if needed
dvc remote add -d myremote /path/to/remote
dvc push
```

**MLflow shows no runs**:
- Make sure you ran training with MLflow installed (it should print `✅ Experiment logged successfully`)
- Start the UI from the project root and point it to this repo's `mlruns/`:
  `python -m mlflow ui --backend-store-uri "file:$(pwd)/mlruns"`
- If port 5000 is busy, use `--port 5001`

**`pip install -r requirements.txt` fails on macOS / Python 3.12+**:
- Recreate the venv and re-run install. This repo pins versions with wheels for Python 3.12.

**Tests Fail**:
```bash
# Run specific test
pytest tests/test_data_preprocessing.py::test_load_data -v

# Debug with print statements
pytest tests/ -s
```

**Drift Monitoring Fails**:
- Install Evidently: `pip install evidently`
- Check that reports/ directory exists

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure CI/CD passes
5. Submit a pull request

## License

This project is for educational purposes as part of the TripleTen MLOps sprint.