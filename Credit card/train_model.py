import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


df = pd.read_csv("dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]


model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)


joblib.dump(model, "model.pkl")

print("Model trained and saved!")