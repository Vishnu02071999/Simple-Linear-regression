# Simple-Linear-regression
Machine Learning project showcasing Simple Linear Regression with feature scaling. The model learns the relationship between calorie consumption and body weight, using StandardScaler for preprocessing and Scikit-learn's LinearRegression for training and prediction.

The project covers:

- Data preprocessing
- Feature scaling using StandardScaler
- Model training using Linear Regression
- Making predictions on unseen data
- Understanding the mathematics behind Linear Regression

---

## Problem Statement

A fitness coach wants to estimate a person's body weight based on the number of calories they consume daily.

Given historical data containing calorie intake and corresponding body weight, our goal is to build a machine learning model capable of predicting weight for new calorie values.

### Dataset

| Calories | Weight (kg) |
|-----------|-------------|
| 1800      | 60          |
| 2000      | 65          |
| 2200      | 70          |
| 2400      | 75          |
| 2600      | 80          |

### Business Question

> Can we predict a person's weight based on their daily calorie intake?

---

## Data Visualization

The relationship between calories consumed and body weight appears approximately linear.

```text
Weight (kg)

80 |                         *
75 |                    *
70 |               *
65 |          *
60 |     *
   +--------------------------------
    1800  2000  2200  2400  2600

          Calories Consumed
```

As calorie intake increases, weight also increases.

This suggests that Linear Regression may be an appropriate model.

---

## Understanding Linear Regression

Linear Regression attempts to find the best-fit straight line through the data points.

### Regression Equation

y = mx + c

Where:

- y = Predicted Weight
- x = Calories Consumed
- m = Slope
- c = Intercept

---

## Step 1: Calculate Mean Values

### Mean Calories

x̄ = (1800 + 2000 + 2200 + 2400 + 2600) / 5

x̄ = 2200

### Mean Weight

ȳ = (60 + 65 + 70 + 75 + 80) / 5

ȳ = 70

---

## Step 2: Calculate the Slope

The slope determines how much the target variable changes for every unit increase in the feature.

Formula:

m = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]

### Calculation Table

| X | Y |  X - x̄  | Y - ȳ   | Product |
|---|---|---------|---------|---------|
|1800|60| -400    |-10      |4000     |
|2000|65| -200    |-5       |1000     |
|2200|70| 0       |0        |0        |
|2400|75| 200     |5        |1000     |
|2600|80| 400     |10       |4000     |

Σ Product = 10000

Σ(X - x̄)² = 400000

Therefore:

m = 10000 / 400000

m = 0.025

---

## Step 3: Calculate the Intercept

Formula:

c = ȳ - mx̄

c = 70 - (0.025 × 2200)

c = 15

---

## Final Regression Equation

Predicted Weight = 0.025 × Calories + 15

---

## Example Prediction

Suppose a person consumes:

Calories = 2300

Using our equation:

Weight = (0.025 × 2300) + 15

Weight = 72.5 kg

### Predicted Weight

72.5 kg

---

## Feature Scaling

Although scaling is not strictly necessary for a dataset with a single feature, it is included to demonstrate a common machine learning preprocessing technique.

### Standardization Formula

z = (x - μ) / σ

Where:

- z = Scaled Value
- μ = Mean
- σ = Standard Deviation

### Why Scale Features?

Feature scaling helps:

- Speed up Gradient Descent
- Improve numerical stability
- Prevent large-valued features from dominating smaller-valued features
- Improve performance of distance-based algorithms such as KNN and SVM

---

## Algorithm Implementation

### Import Libraries

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
```

### Create Dataset

```python
df = pd.DataFrame({
    'Calories': [1800, 2000, 2200, 2400, 2600],
    'Weight': [60, 65, 70, 75, 80]
})
```

### Separate Features and Target

```python
X = df[['Calories']]
y = df['Weight']
```

### Scale Features

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Train Model

```python
model = LinearRegression()
model.fit(X_scaled, y)
```

### Predict New Value

```python
new_calories = [[2300]]

new_scaled = scaler.transform(new_calories)

prediction = model.predict(new_scaled)

print(prediction[0])
```

Output:

```python
72.5
```

---

## Model Workflow

```text
Raw Data
    │
    ▼
Feature Scaling
    │
    ▼
Linear Regression Training
    │
    ▼
Model Learned
    │
    ▼
Prediction
```

---

## Libraries Used

- Python
- Pandas
- Scikit-learn

---

## Key Learnings

- Understanding the mathematics behind Linear Regression
- Calculating slope and intercept manually
- Using StandardScaler for feature preprocessing
- Training a Linear Regression model using Scikit-learn
- Making predictions on unseen data
- Understanding when feature scaling is useful

---
