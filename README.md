# SaaS Subscription Churn Prediction 🚀

An end-to-end machine learning project that predicts whether a SaaS customer is likely to churn, using customer behavior and subscription data. The project includes data preprocessing, model training, evaluation, and a Streamlit-based interactive web dashboard.

---

## 🔍 Problem Statement
Customer churn is a major challenge for SaaS businesses. Retaining customers is significantly cheaper than acquiring new ones. This project predicts customer churn so businesses can take proactive retention actions.

---

## 🧠 Solution Overview
- Built a machine learning pipeline using **Random Forest**
- Handled class imbalance using **SMOTE**
- Performed feature engineering and analysis using **Pandas**
- Created an interactive **Streamlit web dashboard** for predictions

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- SMOTE (imbalanced-learn)
- Streamlit
- Matplotlib & Seaborn
- Git & GitHub

---

## 📊 Features Used
- Gender  
- Location  
- Subscription Plan  
- Monthly Spend  
- Logins in last 30 days  
- Spend per login (engineered feature)

---

## ⚙️ Project Structure
SaaS-Subscription-Churn-Prediction/
│
├── data/ # Dataset
├── images/ # Plots & visualizations
├── models/ # Trained model
├── src/ # Source code
│ ├── churn_model.py
│ ├── app.py
│ └── predict_example.py
├── requirements.txt
└── README.md
