# Weather Condition Classification

A machine-learning project for classifying weather conditions using
temporal, meteorological, and lag-based features.

## Project Overview

This project develops a weather-condition classification pipeline
using historical weather observations. The workflow includes data
preprocessing, temporal feature engineering, lag-feature generation,
feature selection, and comparative evaluation of multiple
classification algorithms.

## Methodology

The project follows these major stages:

1. Data loading and validation
2. Time and date feature engineering
3. Condition preprocessing
4. Lag feature generation
5. Missing-value handling
6. Feature scaling and normalization
7. Target-label encoding
8. Feature selection
9. Model training
10. Model evaluation

## Feature Selection

Multiple feature-selection approaches are applied:

- Variance Threshold
- Correlation Analysis
- Mutual Information
- Lasso

Features selected through these methods are combined to form the
final predictor set.

## Machine Learning Models

The following classifiers are evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Neural Network
- XGBoost

Model performance is evaluated using classification accuracy.

## Repository Structure

```text
weather-condition-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Weather_Condition_Classification_Organized.ipynb
├── data/
├── src/
└── results/
