# ❤️ Heart Disease Risk Prediction System

> An end-to-end machine learning pipeline for heart disease risk prediction, covering data preprocessing, feature engineering, dimensionality reduction, model optimization, and deployment through an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red)
![Healthcare AI](https://img.shields.io/badge/Healthcare-AI-success)

---

# Overview

This project presents a complete machine learning workflow for predicting heart disease using clinical patient data from the UCI Heart Disease dataset.

Rather than training a single model, the repository demonstrates the complete lifecycle of an ML project—from exploratory data analysis and feature engineering to model optimization, deployment, and interactive inference.

The final system predicts whether a patient is at risk of heart disease using **13 clinical attributes** and is deployed through a **Streamlit web application** for real-time prediction.

---

# Features

- End-to-end machine learning workflow
- Exploratory Data Analysis (EDA)
- Data preprocessing & feature engineering
- Missing value handling
- Feature scaling
- PCA dimensionality reduction
- Statistical & model-based feature selection
- Supervised learning
- Unsupervised clustering
- Hyperparameter optimization
- Interactive Streamlit dashboard
- Deployment-ready project structure

---

# System Architecture

```
Clinical Dataset
        │
 Exploratory Data Analysis
        │
 Data Cleaning & Encoding
        │
 Feature Scaling
        │
Feature Engineering
        │
 ┌───────────────┐
 │               │
 ▼               ▼
PCA      Feature Selection
 │               │
 └───────┬───────┘
         ▼
 Supervised Learning
         │
 Hyperparameter Tuning
         │
 Best Model
         │
 Streamlit Deployment
```

---

# Dataset

Source:

**UCI Machine Learning Repository**

| Property | Value |
|-----------|--------|
| Samples | 303 Patients |
| Features | 13 Clinical Variables |
| Task | Binary Classification |
| Classes | Heart Disease / No Heart Disease |

---

# Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Data cleaning
- Feature encoding
- Standardization
- Train/Test split

---

### Dimensionality Reduction

- Principal Component Analysis (PCA)
- Explained variance analysis
- Component visualization

---

### Feature Selection

Implemented multiple feature selection techniques including:

- Chi-Square
- ANOVA F-Test
- Mutual Information
- Random Forest Feature Importance
- Recursive Feature Elimination (RFE)
- L1 Regularization

---

### Supervised Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

---

### Unsupervised Learning

- K-Means
- Hierarchical Clustering
- Silhouette Analysis

---

### Hyperparameter Optimization

- GridSearchCV
- RandomizedSearchCV
- Stratified Cross Validation

---

# Results

| Metric | Value |
|---------|-------|
| Dataset Size | 303 Patients |
| Features | 13 |
| Best Accuracy | **92%** |
| Best F1 Score | **0.91** |
| Best Models | Random Forest / SVM |

The optimized Random Forest and SVM models achieved the highest classification accuracy, demonstrating the effectiveness of feature engineering and hyperparameter optimization on structured clinical data.

---

# Quick Start

## Clone Repository

```bash
git clone https://github.com/motaz2020/Heart-Disease-Risk-Prediction-System.git

cd Heart-Disease-Risk-Prediction-System
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Linux / macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run the ML Pipeline

Execute the notebooks sequentially:

```
01_data_preprocessing.ipynb

02_pca_analysis.ipynb

03_feature_selection.ipynb

04_supervised_learning.ipynb

05_unsupervised_learning.ipynb

06_hyperparameter_tuning.ipynb
```

---

## Launch the Web Application

```bash
streamlit run ui/app.py
```

Open your browser:

```
http://localhost:8501
```

---

# Repository Structure

```
Heart-Disease-Risk-Prediction-System/

│
├── data/
├── notebooks/
├── models/
├── results/
├── ui/
├── deployment/
├── requirements.txt
└── README.md
```

---

# Technology Stack

### Machine Learning

- Scikit-Learn
- XGBoost

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Plotly

### Deployment

- Streamlit

### Utilities

- Joblib
- SciPy
