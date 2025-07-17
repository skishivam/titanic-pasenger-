import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("data/train.csv")

# Preprocessing
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Split
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save model
with open("titanic_model/model.pkl", "wb") as f:
    pickle.dump(clf, f)
