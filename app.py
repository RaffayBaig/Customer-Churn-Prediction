import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load Model & Preprocessor
# ===============================

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/final_model.pkl")
    preprocessor = joblib.load("model/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()


# ===============================
# UI Setup
# ===============================

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title("📊 Customer Churn Prediction System")

st.write("Enter customer details to predict churn probability")

st.divider()


# ===============================
# Input Fields
# ===============================

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 72, 12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)


# ===============================
# Create DataFrame
# ===============================

input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])


# ===============================
# Prediction
# ===============================

if st.button("🔮 Predict Churn"):

    X_processed = preprocessor.transform(input_df)

    prob = model.predict_proba(X_processed)[0][1]
    pred = "Yes" if prob >= 0.5 else "No"

    st.divider()

    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{prob:.2%}")

    if pred == "Yes":
        st.error("⚠️ High Risk of Customer Churn")
    else:
        st.success("✅ Low Risk of Customer Churn")
