import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

print("Loading dataset...")
data = pd.read_csv("creditcard.csv")

print("Preparing data...")
X = data.drop("Class", axis=1)
y = data["Class"]

print("Fixing imbalance...")
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

print("Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Evaluating model...")
pred = model.predict(X_test)
print(classification_report(y_test, pred))

print("Done!")

import pickle

# Assuming your model is called 'model'
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl")
