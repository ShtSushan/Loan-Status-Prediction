import streamlit as st
import numpy as np
import joblib

# ============================================================
# Load saved model and scaler
# ============================================================
model  = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')


# Hide Streamlit UI elements
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="Loan Status Predictor", page_icon="🏦")
st.title("🏦 Loan Status Predictor")
st.markdown("Fill in the details below to check loan approval status.")
st.divider()

# ============================================================
# Input Form
# ============================================================
col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender",           ["Male", "Female"])
    married        = st.selectbox("Married",          ["Yes", "No"])
    dependents     = st.selectbox("Dependents",       [0, 1, 2, 3])
    education      = st.selectbox("Education",        ["Graduate", "Not Graduate"])
    self_employed  = st.selectbox("Self Employed",    ["Yes", "No"])
    credit_history = st.selectbox("Credit History",   [1, 0],
                                   format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")

with col2:
    applicant_income   = st.number_input("Applicant Income (monthly)",   min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income (monthly)", min_value=0, value=0)
    loan_amount        = st.number_input("Loan Amount (in thousands)",   min_value=0, value=100)
    loan_term          = st.selectbox("Loan Term (months)",              [360, 180, 120, 84, 60, 36])
    property_area      = st.selectbox("Property Area",                   ["Semiurban", "Urban", "Rural"])

st.divider()

# ============================================================
# Predict Button
# ============================================================
if st.button("Check Loan Status", use_container_width=True):

    # Encode inputs to match training format
    area_map = {"Semiurban": 0, "Urban": 1, "Rural": 2}

    raw = np.array([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        int(dependents),
        0 if education == "Graduate" else 1,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        area_map[property_area]
    ]], dtype=float)

    # Scale numeric columns (same indices as training: 5,6,7,8)
    raw[:, [5, 6, 7, 8]] = scaler.transform(raw[:, [5, 6, 7, 8]])

    result = model.predict(raw)
    proba  = model.predict_proba(raw)[0]

    if result[0] == 1:
        st.success("✅ Loan Approved!")
        st.metric("Approval Confidence", f"{proba[1]*100:.1f}%")
    else:
        st.error("❌ Loan Not Approved")
        st.metric("Rejection Confidence", f"{proba[0]*100:.1f}%")