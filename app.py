import streamlit as st
import pickle
import numpy as np
import gdown

# ------------------------------
# Download model from Google Drive
# ------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1CszEFu4owf1117WQwaig_77kHe0Ny9Tn"
SCALER_URL = "https://drive.google.com/uc?id=1O2ICsqgW0rN630LMlhKOTtP0TUpBQE8W"

MODEL_FILE = "diabetes_model.pkl"
SCALER_FILE = "scaler.pkl"

gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
gdown.download(SCALER_URL, SCALER_FILE, quiet=False)

# ------------------------------
# Load model & scaler
# ------------------------------
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model/scaler: {e}")

# ------------------------------
# Encoding dicts
# ------------------------------
gender_dict = {"Male": 1, "Female": 0}

smoking_mapping = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5
}

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.write("Fill in your details below and click **Predict** to check your diabetes risk.")

# ------------------------------
# User input fields
# ------------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# ✅ Use Yes/No for hypertension & heart disease
hypertension_label = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease_label = st.selectbox("Heart Disease", ["No", "Yes"])

# ✅ Convert to 0/1 for the model
hypertension = 1 if hypertension_label == "Yes" else 0
heart_disease = 1 if heart_disease_label == "Yes" else 0

smoking_history = st.selectbox(
    "Smoking History",
    ["No Info", "current", "ever", "former", "never", "not current"]
)

bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, format="%.2f")
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=500.0, value=100.0, format="%.2f")

# ------------------------------
# Encode inputs
# ------------------------------
gender_encoded = gender_dict[gender]
smoking_encoded = smoking_mapping[smoking_history]

# ------------------------------
# Prepare & scale input
# ------------------------------
input_data = np.array([[gender_encoded, age, hypertension, heart_disease,
                        smoking_encoded, bmi, blood_glucose_level]])

input_scaled = scaler.transform(input_data)

# ------------------------------
# Predict & display
# ------------------------------
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("⚠️ You may have diabetes. Please consult a doctor.")
    else:
        st.success("✅ You are unlikely to have diabetes. Stay healthy!")

st.caption(" *Always consult a medical professional for an accurate diagnosis.*")
