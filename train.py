import pandas as pd, yaml, joblib, os
from sklearn.ensemble import RandomForestClassifier

p = yaml.safe_load(open("params.yaml"))["train"]
train = pd.read_csv("data/prepared/train.csv")
X, y = train.drop(columns="target"), train["target"]
model = RandomForestClassifier(n_estimators=p["n_estimators"],
                               max_depth=p["max_depth"], random_state=p["seed"])
model.fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")