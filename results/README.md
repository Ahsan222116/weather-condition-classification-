# Model Evaluation Results

## Overview

This directory contains the results and visualizations generated during the weather-condition classification experiment.

Five machine learning approaches and one ensemble boosting model were evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Neural Network
- XGBoost

The models were evaluated using a test set containing **15,324 samples**.

---

## Model Accuracy Comparison

| Model | Accuracy |
|---|---:|
| Logistic Regression | 57.45% |
| Random Forest | 75.68% |
| SVM | 67.20% |
| KNN | 68.27% |
| Neural Network | 67.36% |
| XGBoost | 74.61% |

### Accuracy Visualization

The `model_accuracy_comparison.png` file provides a visual comparison of the classification accuracy achieved by the evaluated models.

The reported results show that:

- Logistic Regression achieved **57.45%** accuracy.
- SVM achieved **67.20%** accuracy.
- Neural Network achieved **67.36%** accuracy.
- KNN achieved **68.27%** accuracy.
- XGBoost achieved **74.61%** accuracy.
- Random Forest achieved **75.68%** accuracy.

Random Forest and XGBoost produced the highest overall accuracy among the evaluated models.

---

## Classification Performance

### Logistic Regression

Logistic Regression achieved an accuracy of approximately **57%**.

Its weighted average precision, recall, and F1-score were approximately:

- Precision: **0.55**
- Recall: **0.57**
- F1-score: **0.56**

The classification report shows considerably weaker performance for several less frequent weather-condition classes. :contentReference[oaicite:1]{index=1}

---

### Random Forest

Random Forest achieved the highest reported accuracy at approximately **76%**.

Performance:

- Accuracy: **0.76**
- Weighted Precision: **0.75**
- Weighted Recall: **0.76**
- Weighted F1-score: **0.74**

The model performed particularly well on several high-support classes. For example, the reported recall for:

- Clear: **0.88**
- Fog: **0.96**
- Haze: **0.93**
- Scattered clouds: **0.71**

The macro-average F1-score was **0.34**, which is substantially lower than the weighted F1-score. This difference indicates that performance varies considerably across classes, particularly for classes with relatively few samples. :contentReference[oaicite:2]{index=2}

---

### Support Vector Machine (SVM)

SVM achieved an accuracy of approximately **67%**.

Performance:

- Accuracy: **0.67**
- Weighted Precision: **0.66**
- Weighted Recall: **0.67**
- Weighted F1-score: **0.65**

The model performed strongly on some high-support classes such as Clear, Fog, and Haze, while several minority classes had very low recall or F1-scores. :contentReference[oaicite:3]{index=3}

---

### K-Nearest Neighbors (KNN)

KNN achieved approximately **68%** accuracy.

Performance:

- Accuracy: **0.68**
- Weighted Precision: **0.67**
- Weighted Recall: **0.68**
- Weighted F1-score: **0.67**

Examples of reported performance include:

- Clear: F1-score **0.74**
- Fog: F1-score **0.87**
- Haze: F1-score **0.72**
- Passing clouds: F1-score **0.62**
- Scattered clouds: F1-score **0.63**

As with the other models, performance was considerably weaker for several low-support classes. :contentReference[oaicite:4]{index=4}

---

### Neural Network

The Neural Network achieved approximately **67%** accuracy.

Performance:

- Accuracy: **0.67**
- Weighted Precision: **0.66**
- Weighted Recall: **0.67**
- Weighted F1-score: **0.66**

The model achieved strong performance on several frequently occurring classes, including:

- Clear: F1-score **0.74**
- Fog: F1-score **0.92**
- Haze: F1-score **0.82**
- Warm: F1-score **0.83**

However, several low-support classes received very low or zero recall. :contentReference[oaicite:5]{index=5}

---

### XGBoost

XGBoost achieved approximately **75%** accuracy.

Performance:

- Accuracy: **0.75**
- Weighted Precision: **0.74**
- Weighted Recall: **0.75**
- Weighted F1-score: **0.73**

The model performed particularly well on several major classes:

- Clear: F1-score **0.79**
- Fog: F1-score **0.94**
- Haze: F1-score **0.87**
- Scattered clouds: F1-score **0.67**
- Sunny: F1-score **1.00**
- Warm: F1-score **0.80**

The macro-average F1-score was **0.41**, compared with a weighted F1-score of **0.73**. :contentReference[oaicite:6]{index=6}

---

# Feature Importance Analysis

XGBoost feature importance was analyzed to identify the input variables contributing most strongly to the model.

The top reported features were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Visibility | 0.244564 |
| 2 | Humidity | 0.140110 |
| 3 | hour_cos | 0.131342 |
| 4 | Wind Speed(km/h) | 0.056921 |
| 5 | hour_sin | 0.036534 |
| 6 | Temperature(°C) | 0.034316 |
| 7 | Lag_1_Pressure(mbar) | 0.033067 |
| 8 | Lag_1_Temperature(°C) | 0.029414 |
| 9 | Pressure(mbar) | 0.027087 |
| 10 | Lag_3_Temperature(°C) | 0.025510 |
| 11 | Lag_3_Visibility | 0.023603 |
| 12 | Lag_1_Visibility | 0.022831 |
| 13 | Day of Week | 0.022644 |
| 14 | Lag_6_Temperature(°C) | 0.021518 |
| 15 | Lag_24_Pressure(mbar) | 0.021078 |

---

## Most Important Features

### 1. Visibility

**Importance: 0.244564**

Visibility is the most important reported feature, accounting for approximately **24.46%** of the total reported feature importance.

This indicates that visibility-related information provides a substantial contribution to the XGBoost model's classification decisions.

### 2. Humidity

**Importance: 0.140110**

Humidity is the second most important feature, with an importance of approximately **14.01%**.

### 3. Hour Cosine Encoding

**Importance: 0.131342**

The `hour_cos` feature has an importance of approximately **13.13%**.

This feature represents the cyclical nature of time and allows the model to capture patterns associated with different times of day.

### 4. Wind Speed

**Importance: 0.056921**

Wind Speed contributes approximately **5.69%** of the reported feature importance.

### 5. Hour Sine Encoding

**Importance: 0.036534**

The `hour_sin` feature contributes approximately **3.65%**.

Together, `hour_sin` and `hour_cos` allow time-of-day information to be represented as a cyclical variable rather than treating hours as independent numerical values.

---

# Feature Importance Visualization

The `feature_selection_comparison.png` file can be used to visualize and compare the contribution of selected features.

The `correlation_heatmap.png` file provides a correlation-based view of relationships among the numerical variables used in the analysis.

The feature-importance results indicate that the model relies heavily on:

1. Visibility
2. Humidity
3. Time-of-day information
4. Wind speed
5. Temperature
6. Historical/lagged weather variables
7. Atmospheric pressure

---

# Model Comparison

The overall accuracy results can be summarized as follows:

```text
Random Forest       75.68%
XGBoost             74.61%
KNN                 68.27%
Neural Network      67.36%
SVM                 67.20%
Logistic Regression 57.45%
