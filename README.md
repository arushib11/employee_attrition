# Employee Attrition: an MLOps Pipeline

An end-to-end MLOps pipeline around a simple attrition classifier. The model is deliberately basic; the project is about everything that surrounds it: data versioning, experiment tracking, automated tests, CI/CD with quality gates, and drift monitoring.

Built for the MLOps sprint of the TripleTen AI/ML Bootcamp.

## What the pipeline does

| Stage | Tool | What happens |
|---|---|---|
| Data versioning | Git + DVC (S3 remote) | The dataset is tracked by a DVC pointer file, not committed to Git |
| Training | scikit-learn | Random Forest, configured in `configs/config.yaml` |
| Experiment tracking | MLflow | Parameters, metrics, the data version and the model are logged for every run |
| Testing | pytest | 16 tests: 8 preprocessing, 6 data and model validation, 2 training and evaluation |
| CI/CD | GitHub Actions | Tests run on every push and pull request; training runs only if tests pass |
| Quality gate | `src/train.py` | Training exits with an error if accuracy or F1 falls below the configured thresholds |
| Drift monitoring | Evidently | Compares training data with simulated production data; exits with an error past a drift threshold |

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset): 1,470 employees, 34 features, binary target (attrition yes/no). About 16% of employees left (237 of 1,470), so the classes are imbalanced.

## Results

Random Forest, 100 trees, max depth 10, 80/20 split (1,176 train, 294 test):

| Metric | Value |
|---|---|
| Accuracy | 0.87 |
| Precision (weighted) | 0.85 |
| F1 (weighted) | 0.82 |

Quality-gate thresholds in `configs/config.yaml`: accuracy ≥ 0.75 and weighted F1 ≥ 0.65. Results can vary slightly with library versions.

### Limitation worth knowing

The metrics above are weighted across both classes, and 84% of employees stay. Weighted scores therefore look healthy even when the model finds few of the people who actually leave. In the classification report that `src/train.py` prints, recall on the attrition class alone is very low (about 0.05 in my run).

That is the main lesson of this project for me: a pipeline can pass every gate and still miss the business question, because the gate measures the wrong thing. The next changes I would make:

- Train with `class_weight="balanced"` or resampling.
- Gate on attrition-class recall or F1, not on weighted averages.
- Tune the decision threshold against the cost of a missed leaver.

## Drift monitoring

`src/monitor_drift.py` builds a simulated production sample (300 rows) with a 10–30% shift in Age, MonthlyIncome, YearsAtCompany and JobSatisfaction, compares it with the training data using Evidently, writes an HTML report to `reports/`, and exits with code 1 if the share of drifted features exceeds 10%. See [MONITORING.md](MONITORING.md) for the analysis.

## Project structure

```
.
├── configs/
│   └── config.yaml                 # Paths, hyperparameters, thresholds
├── data/
│   └── employee_attrition.csv.dvc  # DVC pointer (the CSV itself is not in Git)
├── src/
│   ├── data_preprocessing.py       # Loading, missing values, encoding, split
│   ├── model_training.py           # Training and MLflow logging
│   ├── evaluation.py               # Metrics and classification report
│   ├── train.py                    # Orchestrates the run and applies the quality gate
│   ├── monitor_drift.py            # Evidently drift check
│   └── utils.py
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_data_validation.py
│   └── test_model_training.py
├── .github/workflows/ci-cd.yml     # Test job, then train job
├── run_experiments.py              # Runs six hyperparameter configurations
├── compare_experiments.py          # Compares the logged runs
├── MONITORING.md
└── requirements.txt
```

## Run it

Requires Python 3.9 or later.

```bash
git clone https://github.com/arushib11/employee_attrition.git
cd employee_attrition
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pull the dataset from the public DVC remote
AWS_NO_SIGN_REQUEST=1 dvc pull
```

Train and apply the quality gate:

```bash
PYTHONPATH=$PWD/src python src/train.py
```

Run and compare several configurations:

```bash
python run_experiments.py
python compare_experiments.py
```

Run the tests:

```bash
PYTHONPATH=$PWD/src pytest tests/ -v
```

Check for drift:

```bash
python src/monitor_drift.py
```

Browse experiment runs:

```bash
python -m mlflow ui --backend-store-uri "file:$(pwd)/mlruns"
```

## Design choices

- **Configuration in one file.** Hyperparameters, paths and thresholds live in `configs/config.yaml`, so a run is reproducible from the config and the data version.
- **Tests that need no data download.** The test suite builds a small synthetic sample, so CI can run tests without credentials.
- **Training depends on tests.** The CI training job only starts when the test job passes.
- **Failing loudly.** Both the quality gate and the drift check return a non-zero exit code, which is what lets an automated pipeline stop.

## License

MIT. See [LICENSE](LICENSE).
