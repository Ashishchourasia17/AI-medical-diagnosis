import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/parkinsons.csv")
MODEL_PATH = os.path.join(BASE_DIR, "parkinsons_model.sav")

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please add 'parkinsons.csv' in the 'data' folder.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Print dataset columns for debugging
print("Dataset columns:", df.columns)

# Ensure the dataset has the correct columns
required_columns = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"
]

# Ensure all required columns exist in the dataset
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Missing expected column: {col}")

# Drop non-feature columns
df = df.drop(columns=["name"])  # Remove 'name' as it's not a feature

# Separate features and target
X = df.drop(columns=["status"])  # Features
y = df["status"]  # Target variable

# Check feature count
print("Number of features used in training:", X.shape[1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)

print(f"Parkinson's model saved successfully at: {MODEL_PATH}")
