# Fraud Detection — Midterm Machine Learning

| | |
|---|---|
| Name | Najwa Bilqis Al Khalidah |
| NIM | 101032300186 |
| Class | TK-46-GAB |

---

## Purpose

This repository contains the submission for the Machine Learning midterm assignment. The project builds an end-to-end binary classification pipeline to detect fraudulent financial transactions. The goal is to develop, compare, and track machine learning models that can identify fraud from transaction features, while handling the significant class imbalance inherent to this type of problem.

---

## Project Overview

The dataset (`train_transaction.csv`) contains transaction records with numerical and categorical features describing payment details, card information, and behavioral signals. The target variable `isFraud` is highly imbalanced, with fraud cases representing a small minority of all transactions.

The central challenge is building a model that can reliably identify fraud without excessive false positives, which requires careful handling of class imbalance at both the preprocessing and modeling stages.

---

## Pipeline

```
Load Data (memory-optimized dtypes)
    Preprocessing
        Drop high-missing columns (>75%)
        Feature engineering (log-transform amount, extract hour and day)
        Label encoding for categorical columns
        Median imputation for numeric columns
    Train-Validation Split (80:20, stratified)
    StandardScaler (fit on train only)
    Model Training
        Logistic Regression (baseline, class_weight=balanced)
        XGBoost with Optuna hyperparameter tuning (scale_pos_weight)
    Evaluation (ROC-AUC, PR-AUC, classification report, ROC curve)
    MLflow experiment tracking
    Model persistence
```

---

## Models

**Logistic Regression**
A linear classification baseline using the `saga` solver for efficiency on large datasets. The `class_weight='balanced'` parameter automatically adjusts sample weights to counteract class imbalance. While limited in capturing non-linear patterns, it provides a reliable performance floor and trains quickly.

**XGBoost with Optuna Tuning**
A gradient boosting classifier that builds trees sequentially, with each tree correcting the errors of the previous ensemble. `scale_pos_weight` is set to the ratio of negative to positive samples, which directly addresses class imbalance in the loss function. Hyperparameters were searched over 10 Optuna trials using TPE, maximizing ROC-AUC on the validation set.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Area under the receiver operating characteristic curve — measures the model's ability to separate fraud from non-fraud across all classification thresholds |
| PR-AUC | Area under the precision-recall curve — more informative than ROC-AUC for imbalanced datasets; measures how well the model identifies fraud with high precision at varying recall levels |

Both metrics are computed on the held-out validation set (20% of data). PR-AUC is the primary metric given the class imbalance.

---

## Repository Structure

```
midterm-machine-learning/
    fraud_detection.ipynb    main notebook with full pipeline
    README.md                this file
```

The notebook is organized into 10 sequential cells. Each cell has a markdown header explaining the rationale for that step before the code. Execute cells top to bottom — each cell depends on variables defined in prior cells.

To run the notebook, mount your Google Drive with the dataset at `MyDrive/ML Dataset/train_transaction.csv` and execute in order. Saved model artifacts will be written to `MyDrive/ML Dataset/models_fraud/`.

---

## Tools and Libraries

| Tool | Role |
|------|------|
| scikit-learn | Logistic Regression, preprocessing, metrics |
| XGBoost | Gradient boosting classifier |
| Optuna | Bayesian hyperparameter optimization |
| MLflow | Experiment tracking (parameters, metrics, model artifacts) |
| pandas / numpy | Data manipulation and memory optimization |
| matplotlib | ROC curves and feature importance plots |
| joblib | Model serialization |
