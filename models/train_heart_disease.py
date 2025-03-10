import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/heart_disease.csv")
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.sav")

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please add 'heart_disease.csv' in the 'data' folder.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Preprocess data (Modify as per dataset)
X = df.drop(columns=["target"])  # Features
y = df["target"]  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)

print(f"Heart disease model saved successfully at: {MODEL_PATH}")
