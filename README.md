# Heart Disease Prediction System

> An end-to-end machine learning pipeline for heart disease risk prediction, covering data preprocessing, feature engineering, model optimization, and deployment through an interactive Streamlit application.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red)
![Healthcare AI](https://img.shields.io/badge/Healthcare-AI-success)

---

# Overview

This project implements a complete machine learning workflow for predicting heart disease using clinical patient data from the UCI Heart Disease dataset.

Rather than focusing solely on model training, the project explores the full ML lifecycle—from data preprocessing and feature engineering to model optimization, evaluation, and deployment through an interactive web application.

---

# Features

- End-to-end supervised learning pipeline
- Exploratory Data Analysis (EDA)
- Data preprocessing & feature engineering
- PCA dimensionality reduction
- Statistical and model-based feature selection
- Multiple supervised learning algorithms
- Unsupervised clustering analysis
- Hyperparameter optimization
- Interactive Streamlit application
- Deployment-ready project structure

---

# Machine Learning Pipeline

```
Clinical Dataset
        │
Data Cleaning
        │
Feature Engineering
        │
Scaling
        │
Feature Selection
        │
PCA
        │
Model Training
        │
Hyperparameter Optimization
        │
Evaluation
        │
Streamlit Deployment
```

---

# Dataset

**Source:** UCI Machine Learning Repository

| Property | Value |
|----------|------|
| Samples | 303 |
| Features | 13 Clinical Variables |
| Task | Binary Classification |
| Target | Heart Disease Risk |

---

# Implemented Models

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine

Additional techniques include:

- PCA
- Recursive Feature Elimination (RFE)
- Random Forest Feature Importance
- Chi-Square
- Mutual Information
- GridSearchCV
- RandomizedSearchCV

---

# Results

The optimized pipeline achieved:

- Accuracy: ~83%
- F1 Score: ~0.85
- ROC-AUC: ~0.90

Comparative experiments demonstrated the impact of feature selection and dimensionality reduction on model performance.

---

# Web Application

A Streamlit-based interface allows users to:

- Enter patient clinical information
- Predict heart disease risk
- View prediction probabilities
- Explore model performance metrics

---

# Technologies

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Plotly
- SciPy

---

# Repository Structure

```
Heart-Disease-Prediction
│
├── data
├── notebooks
├── models
├── results
├── ui
├── deployment
├── requirements.txt
└── README.md
```
