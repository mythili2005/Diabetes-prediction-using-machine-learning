import streamlit as st
import pickle
import numpy as np
import gdown

# Download from Google Drive (replace with your own file ID)
url = "https://drive.google.com/file/d/1VsIHHVd6J2qgq0QW_RnnKf4RjsohyQ9s/view?usp=sharing"
output = "fine_tuned_model.pkl"
gdown.download(url, output, quiet=False)

with open(output, "rb") as file:
    model = pickle.load(file)

gender_dict = {"Male": 1, "Female": 0}
smoking_dict = {"Former": 0, "Current": 1, "Never": 2}

st.title("Diabetes Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox("Smoking History", ["Former", "Current", "Never"])
bmi = st.number_input("BMI", min_value=0.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0)

gender_encoded = gender_dict[gender]
smoking_encoded = smoking_dict[smoking_history]

if st.button("Predict"):
    features = np.array([[gender_encoded, age, hypertension, heart_disease, smoking_encoded, bmi, blood_glucose_level]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("You may have diabetes. Please consult a doctor.")
    else:
        st.success("You are not likely to have diabetes.")

