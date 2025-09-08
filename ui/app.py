# In ui/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model and Scaler ---
# Using the correct relative path from the ui directory
try:
    model = joblib.load('../models/final_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'final_model.pkl' and 'scaler.pkl' are in the 'models' directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide", initial_sidebar_state="expanded")


# --- IMPORTANT: This is the core of the fix ---
# This list MUST EXACTLY match the columns from the dataframe after preprocessing in your notebook.
# To get this list, run `print(df.columns.tolist())` in your `01_data_preprocessing.ipynb`
# after all cleaning, encoding, and scaling steps are done.
EXPECTED_COLUMNS = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    'target',  # The target column is in the cleaned CSV, so we include it here and drop it later
    'cp_2.0', 'cp_3.0', 'cp_4.0', 
    'restecg_1.0', 'restecg_2.0',
    'slope_1.0', 'slope_2.0',
    'thal_3.0', 'thal_6.0', 'thal_7.0'
] # Note: This is a likely list, but verify with your notebook!


# --- Helper Function for Preprocessing ---
def preprocess_input(data):
    """Preprocesses user input to match the model's training data format."""
    df = pd.DataFrame([data])
    
    # One-Hot Encode categorical variables
    df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'])

    # Reindex the dataframe to match the training columns, filling missing columns with 0
    # This is a robust way to ensure all required columns are present.
    train_cols = [col for col in EXPECTED_COLUMNS if col != 'target']
    df = df.reindex(columns=train_cols, fill_value=0)

    # Scale numerical features using the loaded scaler
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Ensure the final column order matches the model's expectation
    final_model_features = model.feature_names_in_
    df = df[final_model_features]
    
    return df

# --- Streamlit App UI ---
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("This application uses a Random Forest model to predict the likelihood of heart disease.")

with st.sidebar:
    st.header("Patient Data Input")
    
    age = st.slider("Age", 29, 77, 52)
    sex = st.selectbox("Sex", ("Male", "Female"))
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))
    thalach = st.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", (0, 1, 2, 3, 4))
    
    # --- Corrected Mappings ---
    cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
    restecg = st.selectbox("Resting ECG Results", ("Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"))
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ("Upsloping", "Flat", "Downsloping"))
    thal = st.selectbox("Thalassemia", ("Normal", "Fixed Defect", "Reversible Defect"))

# --- Convert user-friendly input to the correct numerical codes ---
sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0

# These maps now reflect the ACTUAL values in the original dataset
cp_map = {"Typical Angina": 1.0, "Atypical Angina": 2.0, "Non-anginal Pain": 3.0, "Asymptomatic": 4.0}
restecg_map = {"Normal": 0.0, "ST-T wave abnormality": 1.0, "Left ventricular hypertrophy": 2.0}
slope_map = {"Upsloping": 0.0, "Flat": 1.0, "Downsloping": 2.0}
thal_map = {"Normal": 3.0, "Fixed Defect": 6.0, "Reversible Defect": 7.0}

user_input = {
    'age': age,
    'sex': sex_val,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_val,
    'thalach': thalach,
    'exang': exang_val,
    'oldpeak': oldpeak,
    'ca': float(ca),
    'cp': cp_map[cp],
    'restecg': restecg_map[restecg],
    'slope': slope_map[slope],
    'thal': thal_map[thal]
}

# --- Prediction Logic ---
if st.sidebar.button("Predict Disease Risk", type="primary"):
    processed_input = preprocess_input(user_input)
    
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.header("Prediction Result")
    prob_disease = prediction_proba[0][1]
    
    if prediction[0] == 1:
        st.error(f"High Risk of Heart Disease", icon="⚠️")
        st.metric(label="Probability of Disease", value=f"{prob_disease*100:.2f}%")
        st.write("Based on the input data, the model predicts a high likelihood of heart disease.")
    else:
        st.success(f"Low Risk of Heart Disease", icon="✅")
        st.metric(label="Probability of Disease", value=f"{prob_disease*100:.2f}%")
        st.write("The model predicts a low likelihood of heart disease.")