# Step 1: upload your CSV from your local machine
from google.colab import files
uploaded = files.upload()  
# → click the “Choose Files” button and pick your milknew.csv

# Step 2: read it into pandas
import io
import pandas as pd
milk = pd.read_csv(io.StringIO(uploaded['milknew.csv'].decode('utf-8')))

# now your original imports and code
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier   
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

print(milk.head())
print(milk.info())
print(milk.isnull().sum())

# scale numeric columns
scaler = StandardScaler()
milk['Temperature'] = scaler.fit_transform(milk[['Temperature']])
milk['pH']          = scaler.fit_transform(milk[['pH']])

# features & target
X = milk.drop('Grade', axis=1)
y = milk['Grade']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.31, random_state=42
)

# train RF
model = RandomForestClassifier(
    n_estimators=150, random_state=42, max_depth=20, bootstrap=True
)
model.fit(X_train, y_train)

# predict & evaluate
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

# overfitting check
print("\nTESTING FOR OVERFITTING")
print(f"Train accuracy: {model.score(X_train, y_train):.2f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.2f}")

# optional: feature importances
importances = model.feature_importances_
for feat, imp in zip(X.columns, importances):
    print(f"{feat}: {imp:.4f}")
