# Heart Disease Prediction Project

This project implements a comprehensive machine learning pipeline to predict the presence of heart disease using the UCI Heart Disease dataset.

## Project Workflow
1.  **Data Preprocessing**: Cleaned the data, handled missing values, and performed one-hot encoding and feature scaling.
2.  **Dimensionality Reduction**: Analyzed the data structure using PCA.
3.  **Feature Selection**: Used Random Forest feature importance to select the most relevant features for modeling.
4.  **Supervised Learning**: Trained and evaluated several classification models (Logistic Regression, Decision Tree, Random Forest, SVM).
5.  **Unsupervised Learning**: Explored data patterns using K-Means and Hierarchical Clustering.
6.  **Hyperparameter Tuning**: Optimized the best-performing model (Random Forest) using GridSearchCV.
7.  **Deployment**: Built an interactive web UI with Streamlit and deployed it using Ngrok.

## File Structure
```
Heart_Disease_Project/
│── data/
│ ├── heart_disease.csv (raw)
│ ├── cleaned_heart_disease.csv
│ └── selected_features_heart_disease.csv
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│   ... (and so on)
│── models/
│ ├── final_model.pkl
│ └── scaler.pkl
│── ui/
│ └── app.py
│── README.md
└── requirements.txt
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd Heart_Disease_Project
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run ui/app.py
    ```