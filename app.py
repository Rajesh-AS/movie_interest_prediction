import streamlit as st
import pickle
import os   # ✅ THIS LINE WAS MISSING
import numpy as np


st.set_page_config(page_title="Movie Interest Predictor", page_icon="🎬")
st.title("🎬 Movie Interest Prediction")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "movie_interest_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.success("✅ Model loaded successfully")

# Inputs (MATCH TRAINING DATA)
age = st.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])

# Encode gender manually (same logic as LabelEncoder)
gender_encoded = 1 if gender == "Male" else 0

if st.button("Predict"):
    input_data = np.array([[age, gender_encoded]])
    prediction = model.predict(input_data)

    st.subheader("🎯 Prediction Result")
    st.success(f"User Interest: {prediction[0]}")
