import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/lung_cancer.csv")
MODEL_PATH = os.path.join(BASE_DIR, "lung_cancer_model.sav")

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please add 'lung_cancer.csv' in the 'data' folder.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Print column names for debugging
print("Dataset columns:", df.columns)

# Ensure the target column exists
if "LUNG_CANCER" not in df.columns:
    raise KeyError("The target column 'LUNG_CANCER' is not found in the dataset. Check column names and update accordingly.")

# Preprocess data
X = df.drop(columns=["LUNG_CANCER"])  # Features
y = df["LUNG_CANCER"]  # Target variable

# Encode categorical features if any
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':  # Check for non-numeric columns
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)

print(f"Lung cancer model saved successfully at: {MODEL_PATH}")
