import joblib
import pandas as pd

pipe = joblib.load("models/saas_churn_model.joblib")

# Example new customer (use same column names/types as training)
row = pd.DataFrame([{
  "Gender":"Male",
  "Location":"IN",
  "SubscriptionPlan":"Basic",
  "MonthlySpend":25,
  "num_logins_30d":1,
  "spend_per_login": 25 / (1 + 1)   # if you used this feature during training
}])

print("Input row:")
print(row)
try:
    pred = pipe.predict(row)
    print("Churn prediction (0=stay, 1=churn):", int(pred[0]))
except Exception as e:
    print("Prediction error:", e)
