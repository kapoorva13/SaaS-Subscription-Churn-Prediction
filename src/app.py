import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="SaaS Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("models/saas_churn_model.joblib")

# Encoding Maps (must match training)
gender_map = {"Male": 1, "Female": 0}
plan_map = {"Basic": 0, "Pro": 1}
location_map = {"IN": 0, "US": 1}

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("📌 Input Customer Details")
st.sidebar.markdown("Fill the inputs to predict whether a customer will churn.")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
location = st.sidebar.selectbox("Location", ["IN", "US"])
plan = st.sidebar.selectbox("Subscription Plan", ["Basic", "Pro"])
monthly = st.sidebar.number_input("Monthly Spend ($)", min_value=0.0, value=50.0)
logins = st.sidebar.number_input("Logins in Last 30 Days", min_value=0, value=5)

# Encoding inputs
gender_encoded = gender_map[gender]
location_encoded = location_map[location]
plan_encoded = plan_map[plan]

# Feature engineering
spend_per_login = monthly / (logins + 1)

# Prepare row
row = pd.DataFrame([{
    "Gender": gender_encoded,
    "Location": location_encoded,
    "SubscriptionPlan": plan_encoded,
    "MonthlySpend": monthly,
    "num_logins_30d": logins,
    "spend_per_login": spend_per_login
}])

# ---------------------------
# MAIN TITLE
# ---------------------------
st.markdown("<h1 style='text-align: center;'>📊 SaaS Subscription Churn Prediction</h1>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# PREDICTION SECTION
# ---------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Customer Overview")
    st.table(row)

with col2:
    st.subheader("📈 Churn Prediction")

    if st.sidebar.button("Predict Churn"):
        pred = model.predict(row)[0]

        # Probability (if model allows)
        try:
            prob = model.predict_proba(row)[0][1]
        except:
            prob = None

        if pred == 1:
            st.error("❌ Churn Prediction: **YES**")
        else:
            st.success("✅ Churn Prediction: **NO**")

        if prob is not None:
            st.metric("Churn Probability", f"{prob*100:.2f}%")

st.write("---")

# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------
st.subheader("📌 Feature Importance")

try:
    feature_names = ["Gender", "Location", "SubscriptionPlan", "MonthlySpend", "num_logins_30d", "spend_per_login"]
    importances = model.feature_importances_

    # Sort
    sorted_idx = np.argsort(importances)
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_values = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=sorted_values, y=sorted_features, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")

    st.pyplot(fig)

except Exception as e:
    st.warning("Feature importance unavailable for this model.")
    st.caption(str(e))

# ---------------------------
# FOOTER
# ---------------------------
st.write("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
