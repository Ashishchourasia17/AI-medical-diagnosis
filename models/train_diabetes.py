import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Get absolute path to the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/diabetes.csv")

# Check if the file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please check the path or add the dataset.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Preprocess data
X = df.drop(columns=["Outcome"])  # Features
y = df["Outcome"]  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.sav")
with open(MODEL_PATH, "wb") as model_file:
    import pickle
    pickle.dump(model, model_file)

print(f"Model saved successfully at: {MODEL_PATH}")
