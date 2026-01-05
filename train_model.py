import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("movie_interest.csv")

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

X = df[["Age", "Gender"]]
y = df["Interest"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save model
with open("movie_interest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and pickle file saved")
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "movie_model.pkl")

