import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

# --- 1. Load All Saved Artifacts ---

# We use @st.cache_resource to load these only once
@st.cache_resource
def load_artifacts():
    """
    Load all the saved model artifacts from disk.
    """
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        selector = joblib.load('selector.joblib')
        
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
            
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
            
        print("✓ Artifacts loaded successfully")
        return model, scaler, selector, encoders, feature_names

    except FileNotFoundError:
        st.error("ERROR: Model files not found. Please run the Jupyter Notebook to create them.")
        return None, None, None, None, None

model, scaler, selector, encoders, feature_names = load_artifacts()

# Define the original numerical and categorical features
# This helps us build the input form
NUMERICAL_FEATURES = [
    'Study_Hours_per_Week', 'Attendance_Percentage', 'Previous_Sem_Score',
    'Family_Income', 'Sleep_Hours', 'Travel_Time', 'Test_Anxiety_Level',
    'Peer_Influence', 'Motivation_Level', 'Library_Usage_per_Week'
]

CATEGORICAL_FEATURES = [
    'Gender', 'Parental_Education', 'Internet_Access', 'Tutoring_Classes',
    'Sports_Activity', 'Extra_Curricular', 'School_Type', 'Teacher_Feedback'
]

# --- 2. Build the Streamlit UI ---

st.set_page_config(page_title="Student Score Predictor", layout="wide")
st.title("🎓 Student Final Score Predictor")
st.write("Enter the student's details to predict their final score.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

# Dictionary to hold user inputs
input_data = {}

# --- A: Numerical Inputs (in the left column) ---
with col1:
    st.header("Numerical Inputs")
    for feature in NUMERICAL_FEATURES:
        # Use reasonable min/max values based on the dataset's .describe()
        if feature == 'Family_Income':
            input_data[feature] = st.number_input(feature, min_value=10000, max_value=100000, value=50000, step=1000)
        elif feature == 'Attendance_Percentage' or feature == 'Previous_Sem_Score':
             input_data[feature] = st.number_input(feature, min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        elif feature == 'Study_Hours_per_Week':
            input_data[feature] = st.number_input(feature, min_value=0.0, max_value=40.0, value=20.0, step=1.0)
        else:
            # For features like 'Sleep_Hours', 'Motivation_Level', etc.
            input_data[feature] = st.number_input(feature, min_value=0.0, max_value=10.0, value=5.0, step=0.5)

# --- B: Categorical Inputs (in the right column) ---
with col2:
    st.header("Categorical Inputs")
    if encoders: # Only show if encoders loaded
        for feature in CATEGORICAL_FEATURES:
            # Get the saved categories from the encoder for this feature
            options = list(encoders[feature].classes_)
            input_data[feature] = st.selectbox(feature, options=options, index=0)
    else:
        st.warning("Encoder files are missing.")


# --- 3. Prediction Logic ---

if st.button("🚀 Predict Final Score", type="primary", use_container_width=True):
    if model and scaler and selector and encoders and feature_names:
        try:
            # --- A: Create DataFrame from inputs ---
            input_df = pd.DataFrame([input_data])
            
            # --- B: Apply Label Encoding ---
            # Create a copy to avoid changing the displayed data
            processed_df = input_df.copy()
            for feature in CATEGORICAL_FEATURES:
                # Use the saved encoder to transform the string to a number
                encoder = encoders[feature]
                processed_df[f"{feature}_Encoded"] = encoder.transform(processed_df[feature])
                processed_df = processed_df.drop(feature, axis=1) # Drop the original string column
            
            # --- C: Ensure Column Order is Correct ---
            # This is critical! The scaler expects the *exact* order from training.
            try:
                processed_df = processed_df[feature_names]
            except KeyError as e:
                st.error(f"Column mismatch error: {e}. Ensure 'feature_names.json' is correct.")
                st.stop()
            
            # --- D: Apply Scaling and Feature Selection ---
            scaled_data = scaler.transform(processed_df)
            selected_data = selector.transform(scaled_data)
            
            # --- E: Make Prediction ---
            prediction = model.predict(selected_data)
            
            st.success(f"## Predicted Final Score: {prediction[0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model artifacts are not loaded. Cannot predict.")