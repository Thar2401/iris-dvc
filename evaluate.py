import pandas as pd, joblib, json
from sklearn.metrics import accuracy_score, f1_score

model = joblib.load("models/model.pkl")
test = pd.read_csv("data/prepared/test.csv")
X, y = test.drop(columns="target"), test["target"]
pred = model.predict(X)
metrics = {"accuracy": accuracy_score(y, pred),
           "f1": f1_score(y, pred, average="macro")}
json.dump(metrics, open("metrics.json", "w"), indent=2)