import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load dataset
df = pd.read_csv("data/saas_churn.csv")

print("\nData Loaded Successfully!")
print(df.head())

# Encode categorical columns
le = LabelEncoder()
for col in ["Gender", "Location", "SubscriptionPlan"]:
    df[col] = le.fit_transform(df[col].astype(str))

# Feature engineering
df["spend_per_login"] = df["MonthlySpend"] / (df["num_logins_30d"] + 1)

# Prepare features and target
X = df.drop(columns=["Churn", "CustomerID"], errors='ignore')
y = df["Churn"]

X_res, y_res = X, y

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/saas_churn_model.joblib")

print("\nModel saved to models/saas_churn_model.joblib")

# Churn distribution plot
os.makedirs("images", exist_ok=True)
plt.figure()
sns.countplot(x=df["Churn"])
plt.title("Churn Distribution")
plt.savefig("images/churn_distribution.png")
plt.close()

print("Churn Distribution Plot Saved!")
# Feature Importance Plot
import numpy as np

try:
    feature_names = X.columns
    importances = model.feature_importances_

    # Sort by importance
    idx = np.argsort(importances)[::-1]
    sorted_features = feature_names[idx]
    sorted_importances = importances[idx]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sorted_importances, y=sorted_features)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png")
    plt.close()

    print("Feature Importance Plot Saved!")

except Exception as e:
    print("Could not generate feature importance plot:", e)

