import streamlit as st
import pickle
import numpy as np

# Load model
with open("movie_interest_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎬 Movie Interest Prediction App")

age = st.number_input("Enter Age", 1, 100, 25)
gender = st.selectbox("Select Gender", ["Female", "Male"])

# Encoding (same as training)
gender_encoded = 1 if gender == "Male" else 0

if st.button("Predict"):
    result = model.predict([[age, gender_encoded]])
    st.success(f"🎥 Movie Interest: {result[0]}")
