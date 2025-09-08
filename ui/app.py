import streamlit as st
import pandas as pd
import numpy as np
import joblib



# Load the trained model and the scaler
model = joblib.load('../models/final_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# --- Helper Functions ---
# Define the expected columns based on the training data (after one-hot encoding)
# This is crucial for creating the input DataFrame correctly.
# NOTE: Update this list based on the *exact* columns from your `01_data_preprocessing` notebook.
EXPECTED_COLUMNS = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
    'ca', 'cp_1', 'cp_2', 'cp_3', 'restecg_1', 'restecg_2', 'slope_1',
    'slope_2', 'thal_1', 'thal_2', 'thal_3'
]

# The features we selected for the final model
FINAL_MODEL_FEATURES = model.feature_names_in_

def preprocess_input(data):
    """Preprocesses user input for the model."""
    # Create a DataFrame from user input
    df = pd.DataFrame([data])

    # One-Hot Encode categorical variables
    df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

    # Align columns with the training data
    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    # Scale numerical features using the loaded scaler
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Select only the features used by the final model
    df = df[FINAL_MODEL_FEATURES]
    
    return df

# --- Streamlit App UI ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter your health metrics to predict the likelihood of heart disease.")

# Create columns for layout
col1, col2 = st.columns(2)

# --- User Input Fields ---
with col1:
    st.header("Patient Vitals")
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ("Male", "Female"))
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))

with col2:
    st.header("Clinical Measurements")
    oldpeak = st.slider("ST depression induced by exercise", 0.0, 6.2, 1.0)
    ca = st.selectbox("Number of major vessels colored by flourosopy", (0, 1, 2, 3, 4))
    cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
    restecg = st.selectbox("Resting Electrocardiographic Results", ("Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"))
    slope = st.selectbox("Slope of the peak exercise ST segment", ("Upsloping", "Flat", "Downsloping"))
    thal = st.selectbox("Thalassemia", ("Normal", "Fixed defect", "Reversable defect"))

# --- Data Conversion ---
# Convert user-friendly input to numerical values the model expects
sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
restecg_map = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed defect": 2, "Reversable defect": 3} # Note: Check .names file for thal mapping

user_input = {
    'age': age,
    'sex': sex_val,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_val,
    'thalach': thalach,
    'exang': exang_val,
    'oldpeak': oldpeak,
    'ca': ca,
    'cp': cp_map[cp],
    'restecg': restecg_map[restecg],
    'slope': slope_map[slope],
    'thal': thal_map[thal]
}

# --- Prediction Logic ---
if st.button("Predict Heart Disease Risk", type="primary"):
    # Preprocess the input
    processed_input = preprocess_input(user_input)
    
    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    # Display the result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"High Risk of Heart Disease (Probability: {prediction_proba[0][1]*100:.2f}%)")
        st.write("Please consult a healthcare professional for further advice.")
    else:
        st.success(f"Low Risk of Heart Disease (Probability: {prediction_proba[0][0]*100:.2f}%)")
        st.write("This is a good sign, but remember to maintain a healthy lifestyle.")