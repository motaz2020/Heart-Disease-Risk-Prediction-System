# 🫀 Heart Disease Prediction - Complete ML Pipeline

A comprehensive machine learning project for predicting heart disease using the UCI Heart Disease dataset. This project implements a full end-to-end ML pipeline from data preprocessing to deployment with a Streamlit web application.

![Heart Disease Prediction](https://img.shields.io/badge/ML-Heart%20Disease%20Prediction-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Project Overview

This project aims to predict heart disease risk using machine learning techniques. It includes:
- **Comprehensive Data Analysis**: EDA, preprocessing, and feature engineering
- **Dimensionality Reduction**: PCA analysis for feature reduction
- **Feature Selection**: Multiple statistical and ML-based selection methods
- **Model Training**: Supervised learning with multiple algorithms
- **Clustering Analysis**: Unsupervised learning for pattern discovery
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV optimization
- **Web Application**: Interactive Streamlit UI for real-time predictions
- **Deployment**: Ready for cloud deployment with Ngrok support

## 🏗️ Project Structure

```
Heart_Disease_Project/
│
├── data/                           # Data files
│   ├── heart_disease.csv          # Original dataset
│   ├── heart_disease_cleaned.csv  # Cleaned dataset
│   ├── X_scaled.csv               # Scaled features
│   ├── X_top_features.csv         # Selected features
│   ├── X_train.csv, X_test.csv    # Train/test splits
│   └── ...
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── models/                         # Saved models
│   ├── final_optimized_model_*.pkl
│   ├── pca_model.pkl
│   └── kmeans_*.pkl
│
├── results/                        # Analysis results
│   ├── feature_selection_results.csv
│   ├── supervised_learning_results.csv
│   ├── clustering_analysis.json
│   ├── final_optimized_models_info.json
│   └── ...
│
├── ui/                            # Streamlit application
│   └── streamlit_app.py
│
├── deployment/                    # Deployment files
│   └── ngrok_setup.txt
│
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore rules
```

## 📊 Dataset Information

**Source**: UCI Machine Learning Repository - Heart Disease Dataset
- **Samples**: 303 patients
- **Features**: 13 clinical parameters
- **Target**: Binary classification (Heart Disease: Yes/No)
- **Class Distribution**: Approximately balanced

### Features Description:
- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (4 categories)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl
- **restecg**: Resting ECG results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment
- **ca**: Number of major vessels (0-3)
- **thal**: Thalassemia type

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone
cd Heart_Disease_Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download the Heart Disease dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) and save as `data/heart_disease.csv`

### 4. Run the Complete Pipeline
Execute the notebooks in order:
1. `01_data_preprocessing.ipynb` - Data cleaning and EDA
2. `02_pca_analysis.ipynb` - Dimensionality reduction
3. `03_feature_selection.ipynb` - Feature selection
4. `04_supervised_learning.ipynb` - Model training
5. `05_unsupervised_learning.ipynb` - Clustering analysis
6. `06_hyperparameter_tuning.ipynb` - Model optimization

### 5. Launch Web Application
```bash
streamlit run ui/streamlit_app.py
```

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- **Missing Value Handling**: Imputation strategies
- **Feature Encoding**: Categorical variable encoding
- **Data Scaling**: StandardScaler and MinMaxScaler
- **Train-Test Split**: Stratified 80/20 split

### 2. Dimensionality Reduction (PCA)
- **Variance Analysis**: Explained variance ratio
- **Component Selection**: Elbow method and variance threshold
- **Visualization**: 2D and 3D PCA plots
- **Feature Transformation**: Reduced dimensionality datasets

### 3. Feature Selection
- **Statistical Tests**: Chi-square, F-test, Mutual Information
- **ML-based Selection**: Random Forest, XGBoost importance
- **Recursive Feature Elimination (RFE)**: With cross-validation
- **Model-based Selection**: SelectFromModel with L1 regularization

### 4. Supervised Learning Models
- **Logistic Regression**: Linear classification
- **Decision Trees**: Non-linear decision boundaries
- **Random Forest**: Ensemble method
- **Support Vector Machine**: Kernel-based classification

### 5. Unsupervised Learning
- **K-Means Clustering**: Centroid-based clustering
- **Hierarchical Clustering**: Agglomerative clustering
- **Cluster Validation**: Silhouette analysis, elbow method
- **Pattern Discovery**: Cluster interpretation and analysis

### 6. Hyperparameter Optimization
- **Grid Search CV**: Exhaustive parameter search
- **Randomized Search CV**: Random parameter sampling
- **Cross-Validation**: 5-fold stratified CV
- **Performance Metrics**: F1-score, Accuracy, ROC-AUC

## 📈 Model Performance

### Best Model Results:
- **Algorithm**: [Best performing model from tuning]
- **F1 Score**: ~0.85+ (varies based on feature set)
- **Accuracy**: ~0.83+ 
- **ROC AUC**: ~0.90+

### Performance Across Different Feature Sets:
- **Original Features**: All 13 clinical parameters
- **PCA Features**: Reduced dimensionality with 90% variance retention
- **Selected Features**: Top features from selection algorithms

## 🌐 Web Application Features

### Interactive Streamlit UI:
- **Patient Data Input**: User-friendly form for clinical parameters
- **Real-time Predictions**: Instant heart disease risk assessment
- **Risk Visualization**: Probability plots and risk factor analysis
- **Model Information**: Performance metrics and feature importance
- **Educational Content**: Heart health tips and dataset information

### Application Sections:
1. **Patient Information Form**: Input clinical parameters
2. **Prediction Results**: Risk assessment with probability scores
3. **Risk Factor Analysis**: Identification of concerning parameters
4. **Model Details**: Performance metrics and hyperparameters
5. **Health Tips**: Evidence-based recommendations
6. **Data Visualization**: Interactive plots and insights

## 🚀 Deployment Options

### 1. Local Deployment
```bash
streamlit run ui/streamlit_app.py
```

### 2. Cloud Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Platform-as-a-Service deployment
- **AWS/GCP/Azure**: Cloud platform deployment

### 3. Ngrok Tunnel (Development)
```bash
pip install pyngrok
# Run in separate terminal
streamlit run ui/streamlit_app.py
# In another terminal
ngrok http 8501
```

## 📋 Requirements

### Python Version:
- Python 3.8 or higher

### Key Dependencies:
- **Data Science**: pandas, numpy, matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Web Framework**: streamlit
- **Model Persistence**: joblib
- **Visualization**: plotly, matplotlib
- **Statistical Analysis**: scipy

### Development Tools:
- Jupyter Notebook/Lab
- Git for version control
- Virtual environment (recommended)

## 🔧 Configuration

### Environment Setup:
```bash
# Create virtual environment
python -m venv heart_disease_env

# Activate environment
# Windows:
heart_disease_env\Scripts\activate
# macOS/Linux:
source heart_disease_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup:
1. Create `data/` directory
2. Download UCI Heart Disease dataset
3. Save as `heart_disease.csv` in data directory
4. Run preprocessing notebook to generate derived datasets

## 📊 Results and Insights

### Key Findings:
1. **Most Important Features**: chest pain type, maximum heart rate, ST depression
2. **Feature Selection Impact**: Reduced features maintain prediction performance
3. **Model Performance**: Ensemble methods (Random Forest) show best results
4. **Clustering Insights**: Patients naturally group into risk categories
5. **PCA Analysis**: First few components capture most variance

### Clinical Insights:
- Chest pain characteristics are strong predictors
- Exercise-related parameters provide valuable information
- Age and gender play significant roles in risk assessment
- Cholesterol and blood pressure are important risk factors
