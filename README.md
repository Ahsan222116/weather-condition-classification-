# Weather Condition Classification — Machine Learning Project

> An end-to-end multiclass machine-learning project for predicting observed weather conditions from historical meteorological observations.

## 📌 Project Overview

This project builds a complete machine-learning pipeline for **weather condition classification** using historical weather observations.

The goal is to learn the relationship between meteorological measurements and the recorded **`Condition`** label. Instead of training only one algorithm, the project compares multiple classification approaches and evaluates them using overall accuracy as well as macro-averaged metrics.

The project demonstrates practical skills in:

- Data cleaning and validation
- Time and date feature engineering
- Temporal lag-feature engineering
- Cyclical time encoding
- Feature selection
- Multiclass classification
- Class-imbalance handling
- Model comparison
- Model evaluation and interpretation
- Reproducible notebook workflows

---

## 🎯 Objective

### Main Question

**Can current meteorological measurements and historical lagged observations be used to classify the recorded weather condition?**

The project investigates:

1. How much predictive information is contained in current and historical weather measurements?
2. Which features remain useful after multiple feature-selection techniques?
3. How do different machine-learning algorithms perform on the same engineered dataset?
4. Which approaches provide the strongest predictive performance?

---

## 📊 Dataset

The source dataset is an Excel workbook containing:

- **77,341 original observations**
- **10 original columns**

Important variables include:

| Variable | Description |
|---|---|
| `DATE` | Observation date |
| `Time` | Observation time |
| `Condition` | Target weather condition |
| `Temperature(Â°C)` | Temperature |
| `Wind Speed(km/h)` | Wind speed |
| `Humidity` | Humidity |
| `Pressure(mbar)` | Atmospheric pressure |
| `Visibility` | Visibility |

After lag-feature generation and removal of the lag warm-up period, the modeling dataset contains approximately **76k observations**.

---

# 🔄 Machine Learning Workflow

```text
Raw Weather Data
       ↓
Data Validation
       ↓
Date / Time Engineering
       ↓
Chronological Sorting
       ↓
Lag Feature Engineering
       ↓
Cyclical Time Encoding
       ↓
Missing-Value Handling
       ↓
Pressure Cleaning
       ↓
Rare-Class Handling
       ↓
Target Encoding
       ↓
Feature Selection
       ↓
Stratified Train/Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Feature Importance & Error Analysis
```

---

# 🛠️ Feature Engineering

## 1. Time Features

The raw time field is converted into a numeric hour feature.

The project also derives the day of week:

```text
Monday    → 1
Tuesday   → 2
Wednesday → 3
Thursday  → 4
Friday    → 5
Saturday  → 6
Sunday    → 7
```

## 2. Cyclical Time Encoding

Hour is a circular variable. For example, `23:00` and `00:00` are adjacent even though their raw numerical values are far apart.

The project therefore creates:

```text
hour_sin
hour_cos
```

using:

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

This allows models to represent daily patterns more naturally.

## 3. Lag Features

Historical observations are added through lag variables.

Examples:

```text
Lag_1_Temperature(Â°C)
Lag_3_Temperature(Â°C)
Lag_6_Temperature(Â°C)
Lag_24_Temperature(Â°C)

Lag_1_Pressure(mbar)
Lag_24_Pressure(mbar)
Lag_720_Pressure(mbar)

Lag_1_Visibility
Lag_3_Visibility
```

The maximum lag used is:

```text
720 observations
```

The dataset is sorted chronologically before shifting so lag values refer to earlier observations.

> **Note:** a lag of 720 means 720 records, not automatically 720 clock-hours. Its real-world time meaning depends on the source sampling frequency.

---

# 🧹 Data Cleaning

## Missing Values

The preprocessing pipeline uses:

- interpolation
- forward filling
- backward filling
- final numeric fallback where necessary

## Pressure Cleaning

Pressure values equal to:

```text
0 mbar
```

are treated as invalid measurements.

They are replaced, interpolated, filled where required, and min-max scaled.

## Redundant Columns

After temporal features have been extracted, redundant metadata fields are removed where present:

```text
Time
date
DATE
Day
Year
Month
```

---

# 🎯 Target Preparation

The `Condition` column is the classification target.

The dataset contains many weather-condition categories, including examples such as:

```text
Clear.
Sunny.
Rain.
Fog.
Haze.
Overcast.
Partly cloudy.
Thunderstorms.
```

## Singleton Classes

The completed experiment identified **three classes with only one observation each**.

A class with one observation cannot appear in both training and testing partitions during stratified evaluation. Therefore, the revised modeling experiment removes only those singleton classes before the train/test split.

This allows a valid:

```python
train_test_split(..., stratify=y)
```

while keeping the decision transparent in the notebook.

---

# 🔎 Feature Selection

Four approaches are used.

## 1. Variance Threshold

Removes features with very low variation.

Configured threshold:

```text
0.01
```

## 2. Correlation Screening

Pearson correlation with the encoded target is used as a screening heuristic.

Threshold:

```text
|correlation| > 0.10
```

Because the target is multiclass, this is treated only as a simple screening method rather than a complete dependence measure.

## 3. Mutual Information

Mutual information is used to capture potentially nonlinear relationships between predictors and the target.

Threshold:

```text
Mutual Information > 0.01
```

## 4. Lasso

Lasso is retained as part of the original project methodology.

The predictor variables are standardized before L1 regularization.

---

# 🤖 Machine Learning Models

The project compares six algorithms:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear multiclass baseline |
| Random Forest | Bagged decision-tree ensemble |
| SVM | Margin-based classifier |
| KNN | Distance-based classifier |
| Neural Network | Nonlinear multilayer model |
| XGBoost | Gradient-boosted tree model |

All models use the same final train/test partitions for a fair comparison.

---

# 🐛 XGBoost Problem and Fix

The original experiment produced:

```text
Invalid classes inferred from unique values of y
```

This occurred because the target labels became non-contiguous after the rare-class issue.

XGBoost multiclass training expects labels such as:

```text
0, 1, 2, 3, ..., n-1
```

The revised notebook fixes this by:

1. Removing singleton classes before splitting
2. Re-encoding the remaining target labels
3. Ensuring class IDs are contiguous
4. Setting the correct `num_class` value

XGBoost can therefore run as a proper multiclass classifier.

---

# 📈 Evaluation

Because the target has multiple classes and is not perfectly balanced, the project reports several metrics.

### Accuracy

Overall proportion of correctly classified observations.

### Macro Precision

Average precision across all classes, with every class receiving equal weight.

### Macro Recall

Average recall across all classes.

### Macro F1

Harmonic mean of macro precision and macro recall.

### Confusion Matrix

Shows which weather conditions are frequently confused.

### Classification Report

Provides precision, recall, and F1 for each class.

---

# 📊 Earlier Baseline Results

The earlier completed run produced:

| Model | Accuracy |
|---|---:|
| Logistic Regression | **51.00%** |
| Random Forest | **70.00%** |
| SVM | **58.69%** |
| KNN | **59.71%** |
| Neural Network | **59.96%** |
| XGBoost | Failed because of non-contiguous class labels |

These values are kept as a baseline reference.

The revised notebook performs a new experiment after:

- singleton-class handling
- valid stratified splitting
- feature-selection refinement
- corrected XGBoost class encoding
- improved model configuration

The final current-run results should be taken from the executed notebook rather than hard-coded into this README.

---

# 📁 Recommended Repository Structure

```text
weather-condition-classification/
│
├── Weather_Condition_Classification_Portfolio_Complete.ipynb
├── Weather_Condition_Classification_Tuned.py
├── README.md
│
├── data/
│   └── Projectdata.xlsx
│
├── outputs/
│   ├── weather_model_results.csv
│   ├── weather_selected_features.csv
│   └── best_weather_model.joblib
│
└── images/
    ├── model_comparison.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

> Avoid committing private, very large, or restricted datasets to GitHub. If the raw data cannot be distributed, document its source and required schema instead.

---

# 💻 Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib

### Machine Learning
- Scikit-learn
- XGBoost

### Environment
- Jupyter Notebook
- Google Colab

---

# ▶️ How to Run

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/weather-condition-classification.git
cd weather-condition-classification
```

## 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost python-dateutil openpyxl joblib
```

## 3. Add the dataset

Place the Excel dataset in the expected location or update:

```python
DATA_PATH = "path/to/Projectdata.xlsx"
```

## 4. Open the notebook

```bash
jupyter notebook
```

Or upload the notebook to Google Colab.

## 5. Run all cells

The notebook will:

- load and validate the dataset
- engineer temporal and lag features
- clean the data
- handle singleton target classes
- perform feature selection
- create a stratified split
- train six models
- evaluate the models
- generate comparison visualizations
- analyze feature importance
- export model artifacts

---

# 🚀 Key Skills Demonstrated

```text
Python
Pandas
NumPy
Data Cleaning
Exploratory Data Analysis
Feature Engineering
Time-Series Features
Lag Features
Cyclical Encoding
Feature Selection
Multiclass Classification
Scikit-learn
XGBoost
Model Comparison
Class Imbalance
Model Evaluation
Confusion Matrices
Feature Importance
Jupyter / Google Colab
```

---

# ⚠️ Limitations

This is a practical machine-learning portfolio project rather than a production weather-forecasting system.

### 1. Feature-selection leakage

The current workflow performs feature selection before the final train/test split to remain consistent with the implemented project methodology.

A stricter research/production pipeline should fit learned feature-selection steps on training data only.

### 2. Lasso with an encoded multiclass target

Lasso is retained because it is part of the project's original methodology.

A classification-specific L1 Logistic Regression selector would be more principled.

### 3. Pressure scaling

Pressure scaling follows the project preprocessing workflow.

For production use, learned transformations should be fitted only on training data and then applied to validation/test data.

### 4. Accuracy alone is not enough

For this multiclass problem, macro precision, recall, F1, and class-level performance should be considered alongside accuracy.

### 5. Lag interpretation

Lag numbers refer to previous observations. Their exact time interval depends on the dataset's sampling frequency.

---

# 🔮 Future Improvements

Potential next steps include:

- Training-only preprocessing pipelines
- Cross-validation
- Hyperparameter optimization
- Classification-specific L1 feature selection
- Better rare-class strategies
- Probability calibration
- Deeper error analysis
- SHAP-based model explainability
- Streamlit or Gradio deployment
- REST API for model inference
- Experiment tracking

---

# 💼 Portfolio Value

This project demonstrates the complete practical machine-learning workflow:

> **Raw data → preprocessing → feature engineering → feature selection → model training → evaluation → debugging → interpretation → export**

It can be presented as a portfolio project for entry-level roles involving:

- Data Science
- Machine Learning
- Data Analytics
- Python
- Predictive Modeling

---

# 👤 Author

**Ahsan Khalil**

**Focus:** Data Science · Machine Learning · AI

This project was developed as part of a practical machine-learning portfolio focused on building reproducible, understandable, and end-to-end data science workflows.

---

## License

Add the license appropriate for your repository.

For example:

```text
MIT License
```

Only use a license that is appropriate for the source code, dataset, and any third-party materials included in the repository.
