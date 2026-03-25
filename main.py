# ================================
# CREDIT CARD FRAUD DETECTION
# ================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ================================
# 2. Load Dataset
# ================================
# 👉 Make sure creditcard.csv is in same folder as this file

data = pd.read_csv("creditcard.csv")

print("First 5 rows:")
print(data.head())

# ================================
# 3. Check Data Info
# ================================
print("\nDataset Info:")
print(data.info())

print("\nClass Distribution:")
print(data['Class'].value_counts())

# ================================
# 4. Visualize Imbalance
# ================================
sns.countplot(x='Class', data=data)
plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
plt.show()

# ================================
# 5. Prepare Data
# ================================
X = data.drop('Class', axis=1)
y = data['Class']

# ================================
# 6. Handle Imbalance (SMOTE)
# ================================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(pd.Series(y_res).value_counts())

# ================================
# 7. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# ================================
# 8. Train Model (XGBoost)
# ================================
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# ================================
# 9. Predictions
# ================================
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# ================================
# 10. Threshold Tuning (Unique Part ⭐)
# ================================
y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.3  # you can change this
y_pred_custom = (y_prob > threshold).astype(int)

print("\nCustom Threshold (0.3) Report:")
print(classification_report(y_test, y_pred_custom))

# ================================
# 11. Feature Importance (Extra ⭐)
# ================================
importances = model.feature_importances_

plt.figure(figsize=(10,5))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.show()

# ================================
# 12. Save Model
# ================================
import pickle
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved as model.pkl ✅")

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()