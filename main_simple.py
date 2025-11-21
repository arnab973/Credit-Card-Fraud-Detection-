import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle

print("Loading dataset...")
data = pd.read_csv("creditcard.csv")

print("Selecting top 7 features...")
# Based on feature selection: V3, V7, V10, V12, V14, V16, V17
selected_features = ['V3', 'V7', 'V10', 'V12', 'V14', 'V16', 'V17', 'Time', 'Amount']
X = data[selected_features]
y = data["Class"]

print("Fixing imbalance...")
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

print("Training simplified model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating model...")
pred = model.predict(X_test)
print(classification_report(y_test, pred))

print("Saving simplified model...")
pickle.dump(model, open("model_simple.pkl", "wb"))
print("Simplified model saved as model_simple.pkl")

print("Feature importance:")
feature_importance = dict(zip(selected_features, model.feature_importances_))
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")
