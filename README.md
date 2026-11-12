# Airline Passenger Satisfaction

Exploring airline passenger satisfaction with classical machine learning, gradient boosting, and a neural network. This project compares several binary classifiers using **ROC-AUC**, then examines tuning results and model blends.

The complete analysis is in [`data.ipynb`](data.ipynb), including saved tables, plots, and experiment outputs.

## Dataset

The local dataset was downloaded for [Kaggle Playground Series — Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10).

| Split | Rows | Columns |
| --- | ---: | ---: |
| Training | 699,635 | 23, including the target |
| Test | 299,844 | 22 |

The target is `satisfaction`: `True` for satisfied passengers and `False` otherwise. Predictors describe passenger demographics, customer and travel categories, flight distance, service ratings, and departure/arrival delays. The `id` column is excluded from model features.

Training labels are approximately **44.4% satisfied** and **55.6% not satisfied**.

Dataset CSVs are not included in this repository. Download them from the competition's Data page, following Kaggle's access requirements and competition rules.

## What's in the notebook

- Exploratory analysis of numerical correlations and satisfaction rates across passenger categories.
- A stratified 80/20 train/validation split with `random_state=42`.
- Median imputation and standard scaling for numerical features; imputation and one-hot encoding for categorical features in the scikit-learn pipelines.
- Logistic Regression, Random Forest, XGBoost, and MLP baselines.
- Random Forest and neural-network tuning, plus a randomized cross-validation search for XGBoost.
- CatBoost experiments using native categorical features and an internal split for early stopping.
- Training/validation AUC comparisons and simple probability blends.

## Recorded results

These are **saved local validation outputs from the notebook**, not Kaggle leaderboard scores or results of an independently rerun benchmark.

| Experiment | Validation ROC-AUC |
| --- | ---: |
| Logistic Regression | 0.92046 |
| Random Forest baseline | 0.95583 |
| XGBoost baseline / randomized-search winner | 0.95717 |
| MLP baseline | 0.95591 |
| Tuned MLP `(128, 64, 32)` | 0.95640 |
| CatBoost runs | ~0.95771–0.95776 |
| CatBoost probability blend | 0.95778 |

Tree ensembles outperform the linear baseline in these recorded experiments. Increasing neural-network size did not produce a substantial improvement, and the reported blend improvement is very small.

### Evaluation caveats

The notebook is an exploratory work log. Some saved outputs reflect different interactive runs and may not match a fresh sequential execution. In particular, `cb_depth6 = cb_model` creates an alias rather than an independent model, so reproducing a blend requires separately trained model instances. Some later comments and outputs also reflect earlier parameter choices.

The same validation split was inspected across multiple experiments. Small AUC differences should therefore be treated as exploratory; they do not establish a reliable generalization gain. The notebook does not demonstrate an AUC of 0.963.

## Run locally

Use Python 3.9 or newer; the notebook metadata records Python 3.9.13. Install its dependencies in your preferred environment:

```bash
python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn xgboost catboost
```

Arrange the downloaded files relative to the notebook:

```text
.
├── README.md
├── data.ipynb
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

Start Jupyter from the repository directory:

```bash
python -m jupyter notebook data.ipynb
```

Review the execution-state caveats above before running all cells. The model-training and randomized-search cells can take considerable time on a CPU. Exact results may vary with package versions and hardware; this repository does not provide a locked environment.

## Next experiments

- Train each candidate with a distinct model object and record its exact configuration.
- Use out-of-fold predictions to compare models and choose ensemble weights.
- Test service-rating encodings and feature interactions with controlled comparisons.
- Keep a separate final evaluation split and report uncertainty for small AUC gains.

## Repository scope

This repository contains the original notebook and this README. Local CSVs, trained model files, submissions, logs, and separate training scripts are excluded.
